"""Scale up Qwen3-0.6B LoRA training to find GPU memory limits.

Tests combinations of:
  - batch_size: 1, 2, 4, 8, 16
  - max_seq_len: 128, 256, 512, 1024, 2048
  - lora_rank: 8, 16, 32, 64
  - num_workers: 1, 2 (single GPU vs both GPUs)
  - gradient_accumulation_steps: 1, 4, 8

Runs each config for a few steps, logs peak GPU memory, reports OOM boundary.
"""

import os
import sys
import json
import time
import itertools
import subprocess
import traceback

import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

OUTPUT_DIR = "/data/outputs/qwen3-scaling"
MODEL_NAME = "Qwen/Qwen3-0.6B"
RESULTS_FILE = "/data/outputs/qwen3-scaling/results.jsonl"


def train_loop_per_worker(config):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from ray.train.huggingface.transformers import (
        RayTrainReportCallback,
        prepare_trainer,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_size = config["batch_size"]
    max_seq_len = config["max_seq_len"]
    lora_rank = config["lora_rank"]
    grad_accum = config["grad_accum"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.2f}%)")

    # Generate enough data for a few steps
    num_samples = max(batch_size * grad_accum * 6, 32)
    texts = ["This is a sample text for scaling test. " * 10] * num_samples

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="/tmp/qwen3-scale-run",
        max_steps=5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[RayTrainReportCallback()],
    )
    trainer = prepare_trainer(trainer)
    trainer.train()

    # Report peak memory
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(f"  Peak allocated: {peak_mem_mb:.0f} MB, "
          f"Peak reserved: {peak_reserved_mb:.0f} MB")


def run_config(cfg, run_id):
    """Run a single config and return result dict."""
    label = (f"bs={cfg['batch_size']} seq={cfg['max_seq_len']} "
             f"r={cfg['lora_rank']} ga={cfg['grad_accum']} "
             f"workers={cfg['num_workers']}")
    print(f"\n{'='*60}")
    print(f"[{run_id}] {label}")
    print(f"{'='*60}")

    result = {**cfg, "run_id": run_id, "status": "unknown"}
    start = time.time()

    try:
        trainer = TorchTrainer(
            train_loop_per_worker,
            train_loop_config=cfg,
            scaling_config=ScalingConfig(
                num_workers=cfg["num_workers"],
                use_gpu=True,
            ),
            run_config=RunConfig(
                name=f"scale-{run_id}",
                storage_path=OUTPUT_DIR,
                failure_config=FailureConfig(max_failures=0),
                checkpoint_config=CheckpointConfig(num_to_keep=1),
            ),
        )
        fit_result = trainer.fit()
        elapsed = time.time() - start

        # Extract metrics from last reported result
        metrics = fit_result.metrics or {}
        result["status"] = "OK"
        result["elapsed_s"] = round(elapsed, 1)
        result["train_loss"] = metrics.get("train_loss")
        result["train_steps_per_second"] = metrics.get("train_steps_per_second")
        # Effective batch = per_device * grad_accum * num_workers
        eff_bs = cfg["batch_size"] * cfg["grad_accum"] * cfg["num_workers"]
        result["effective_batch_size"] = eff_bs
        result["tokens_per_step"] = eff_bs * cfg["max_seq_len"]
        print(f"  -> OK in {elapsed:.1f}s  "
              f"(eff_bs={eff_bs}, tokens/step={eff_bs * cfg['max_seq_len']})")

    except Exception as e:
        elapsed = time.time() - start
        err_str = str(e)
        is_oom = "CUDA out of memory" in err_str or "OutOfMemoryError" in err_str
        result["status"] = "OOM" if is_oom else "ERROR"
        result["elapsed_s"] = round(elapsed, 1)
        result["error"] = err_str[:200]
        print(f"  -> {result['status']} in {elapsed:.1f}s: {err_str[:120]}")

    return result


def main():
    ray.init()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: Single GPU - find batch_size x seq_len limits at r=8
    configs_phase1 = []
    for bs, seq in itertools.product([1, 2, 4, 8, 16], [128, 256, 512, 1024, 2048]):
        configs_phase1.append({
            "batch_size": bs, "max_seq_len": seq,
            "lora_rank": 8, "grad_accum": 1, "num_workers": 1,
        })

    # Phase 2: Scale LoRA rank at best batch_size/seq_len from phase 1
    configs_phase2 = []
    for rank in [16, 32, 64]:
        for bs in [2, 4, 8]:
            configs_phase2.append({
                "batch_size": bs, "max_seq_len": 512,
                "lora_rank": rank, "grad_accum": 1, "num_workers": 1,
            })

    # Phase 3: Dual GPU
    configs_phase3 = []
    for bs in [2, 4, 8, 16]:
        for seq in [512, 1024, 2048]:
            configs_phase3.append({
                "batch_size": bs, "max_seq_len": seq,
                "lora_rank": 8, "grad_accum": 1, "num_workers": 2,
            })

    # Phase 4: Gradient accumulation for larger effective batch
    configs_phase4 = []
    for ga in [4, 8, 16]:
        for bs in [2, 4]:
            configs_phase4.append({
                "batch_size": bs, "max_seq_len": 1024,
                "lora_rank": 8, "grad_accum": ga, "num_workers": 2,
            })

    all_configs = (
        [("Phase 1: Single GPU bs×seq sweep (r=8)", configs_phase1)]
        + [("Phase 2: LoRA rank scaling (seq=512)", configs_phase2)]
        + [("Phase 3: Dual GPU scaling", configs_phase3)]
        + [("Phase 4: Gradient accumulation", configs_phase4)]
    )

    all_results = []
    run_id = 0
    oom_seqs = set()  # Track seq_lens that OOM at bs=1 (skip larger bs)

    with open(RESULTS_FILE, "w") as f:
        for phase_name, configs in all_configs:
            print(f"\n{'#'*60}")
            print(f"# {phase_name}")
            print(f"{'#'*60}")

            for cfg in configs:
                # Skip configs that will obviously OOM
                # (if smaller batch already OOMed at same seq_len)
                skip_key = (cfg["max_seq_len"], cfg["num_workers"], cfg["lora_rank"])
                if skip_key in oom_seqs and cfg["batch_size"] > 1:
                    print(f"\n[{run_id}] SKIP (previous OOM at seq={cfg['max_seq_len']})")
                    run_id += 1
                    continue

                result = run_config(cfg, run_id)
                result["phase"] = phase_name
                all_results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()

                if result["status"] == "OOM" and cfg["batch_size"] <= 2:
                    oom_seqs.add(skip_key)

                run_id += 1

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SCALING RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'bs':>4} {'seq':>5} {'rank':>4} {'ga':>3} {'wrk':>3} "
          f"{'eff_bs':>6} {'tok/step':>9} {'status':>6} {'time':>6} {'loss':>8} {'steps/s':>7}")
    print("-" * 80)

    for r in all_results:
        eff_bs = r.get("effective_batch_size", "")
        tok = r.get("tokens_per_step", "")
        loss = r.get("train_loss", "")
        sps = r.get("train_steps_per_second", "")
        print(f"{r['batch_size']:>4} {r['max_seq_len']:>5} {r['lora_rank']:>4} "
              f"{r['grad_accum']:>3} {r['num_workers']:>3} "
              f"{str(eff_bs):>6} {str(tok):>9} "
              f"{r['status']:>6} {r.get('elapsed_s',''):>6} "
              f"{str(loss)[:8]:>8} {str(sps)[:7]:>7}")

    # Find the max successful config
    ok_results = [r for r in all_results if r["status"] == "OK"]
    if ok_results:
        best_throughput = max(ok_results, key=lambda r: r.get("tokens_per_step", 0))
        print(f"\nMax tokens/step: {best_throughput['tokens_per_step']} "
              f"(bs={best_throughput['batch_size']} seq={best_throughput['max_seq_len']} "
              f"r={best_throughput['lora_rank']} workers={best_throughput['num_workers']} "
              f"ga={best_throughput['grad_accum']})")

    ray.shutdown()


if __name__ == "__main__":
    main()
