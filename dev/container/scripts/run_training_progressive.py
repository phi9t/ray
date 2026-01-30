"""Progressive Qwen3 LoRA training: 0.6B -> 1.7B -> 4B -> 8B.

Trains each model size with QLoRA (4-bit NF4) + gradient checkpointing + 8-bit
paged optimizer on 2x GTX 1070 Ti (8GB each). The 8B model uses device_map="auto"
to split layers across both GPUs.

Each training run executes in a subprocess to:
  1. Control CUDA_VISIBLE_DEVICES (single GPU for small models, both for 8B)
  2. Isolate CUDA context — OOM/errors don't corrupt the parent process
  3. Prevent HF Trainer from wrapping bnb 4-bit models in DataParallel
     (Trainer wraps when n_gpu > 1; bnb quantized tensors can't be replicated)
"""

import json
import os
import subprocess
import sys
import textwrap
import time

OUTPUT_DIR = "/data/outputs/qwen3-scaling"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "progressive_results.jsonl")

MODEL_CONFIGS = [
    {
        "model_name": "Qwen/Qwen3-0.6B",
        "label": "0.6B",
        "seq_len": 2048,
        "batch_size": 1,
        "grad_accum": 1,
        "lora_r": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "cuda_devices": "0",
    },
    {
        "model_name": "Qwen/Qwen3-1.7B",
        "label": "1.7B",
        "seq_len": 2048,
        "batch_size": 1,
        "grad_accum": 1,
        "lora_r": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "cuda_devices": "0",
    },
    {
        "model_name": "Qwen/Qwen3-4B",
        "label": "4B",
        "seq_len": 1024,
        "batch_size": 1,
        "grad_accum": 2,
        "lora_r": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "cuda_devices": "0",
    },
    {
        "model_name": "Qwen/Qwen3-8B",
        "label": "8B",
        "seq_len": 1024,
        "batch_size": 1,
        "grad_accum": 4,
        "lora_r": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "cuda_devices": "0,1",
    },
]

# Probe configs: (seq_len, batch_size)
PROBE_CONFIGS = [
    (512, 2),
    (512, 4),
    (1024, 2),
    (1024, 4),
    (2048, 1),
    (2048, 2),
    (4096, 1),
]

# Worker script executed in subprocess
WORKER_SCRIPT = textwrap.dedent(r'''
import gc
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

config = json.loads(sys.argv[1])

model_name = config["model_name"]
seq_len = config["seq_len"]
batch_size = config["batch_size"]
grad_accum = config["grad_accum"]
lora_r = config["lora_r"]
lora_targets = config["lora_targets"]
max_steps = config["max_steps"]

result = {
    "model": config["label"],
    "model_name": model_name,
    "seq_len": seq_len,
    "batch_size": batch_size,
    "grad_accum": grad_accum,
    "lora_r": lora_r,
    "max_steps": max_steps,
    "status": "unknown",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_visible_gpus": torch.cuda.device_count(),
}

start = time.time()

try:
    # Load model with QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpu = torch.cuda.device_count()
    device_map = "auto" if n_gpu > 1 else {"": 0}
    print(f"  Loading {model_name} (4-bit NF4, {n_gpu} GPU(s), device_map={device_map})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map=device_map, torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r,
        lora_alpha=lora_r * 2, lora_dropout=0.05,
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Log device placement
    devices = set()
    for _, p in model.named_parameters():
        devices.add(str(p.device))
    result["devices_used"] = sorted(devices)
    print(f"  Param devices: {sorted(devices)}")

    # Create dataset
    texts = ["Ray is a unified framework for scaling AI applications. " * 20] * 64
    def tokenize_fn(examples):
        out = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=seq_len)
        out["labels"] = out["input_ids"].copy()
        return out
    dataset = Dataset.from_dict({"text": texts}).map(tokenize_fn, batched=True, remove_columns=["text"])

    # Train
    print(f"  Training {max_steps} steps (bs={batch_size}, ga={grad_accum}, seq={seq_len})")
    training_args = TrainingArguments(
        output_dir="/tmp/qwen3-progressive-run",
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    train_result = trainer.train()
    metrics = train_result.metrics

    elapsed = time.time() - start

    # Memory stats
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.max_memory_allocated(i) / 1024**3
        reserved = torch.cuda.max_memory_reserved(i) / 1024**3
        result[f"gpu{i}_peak_alloc_gb"] = round(alloc, 2)
        result[f"gpu{i}_peak_reserved_gb"] = round(reserved, 2)

    result["status"] = "OK"
    result["elapsed_s"] = round(elapsed, 1)
    result["train_loss"] = metrics.get("train_loss")
    result["train_steps_per_second"] = metrics.get("train_steps_per_second")
    effective_bs = batch_size * grad_accum
    result["effective_batch_size"] = effective_bs
    result["tokens_per_step"] = effective_bs * seq_len

    print(f"  -> OK in {elapsed:.1f}s | loss={result['train_loss']:.4f} "
          f"| {result['train_steps_per_second']:.2f} steps/s")
    for k, v in result.items():
        if "peak_alloc" in k:
            print(f"     {k}: {v} GB")

except Exception as e:
    elapsed = time.time() - start
    err = str(e)
    is_oom = "CUDA out of memory" in err or "OutOfMemoryError" in err
    result["status"] = "OOM" if is_oom else "ERROR"
    result["elapsed_s"] = round(elapsed, 1)
    result["error"] = err[:500]
    print(f"  -> {result['status']} in {elapsed:.1f}s: {err[:200]}")
    if not is_oom:
        traceback.print_exc()

# Output result as JSON on the last line (parent parses this)
print(f"\n__RESULT_JSON__:{json.dumps(result)}")
''')


def run_training_subprocess(config, timeout=600):
    """Run a single training config in a subprocess.

    Returns the result dict parsed from subprocess output.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]
    env["PYTHONUNBUFFERED"] = "1"

    config_json = json.dumps(config)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", WORKER_SCRIPT, config_json],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Print subprocess output in real time (it's captured, so print it)
    if proc.stdout:
        for line in proc.stdout.split("\n"):
            if not line.startswith("__RESULT_JSON__:"):
                print(line)

    if proc.stderr:
        # Print only important stderr lines (skip warnings)
        for line in proc.stderr.split("\n"):
            if any(kw in line for kw in ["Error", "error", "OOM", "CUDA", "Traceback"]):
                print(f"  stderr: {line}")

    # Parse result from last line
    for line in reversed(proc.stdout.split("\n")):
        if line.startswith("__RESULT_JSON__:"):
            return json.loads(line[len("__RESULT_JSON__:"):])

    # If no result parsed, create error result
    return {
        "model": config.get("label", "?"),
        "status": "SUBPROCESS_ERROR",
        "error": f"exit={proc.returncode}, stderr={proc.stderr[-300:] if proc.stderr else 'none'}",
        "elapsed_s": 0,
        **{k: config[k] for k in ["seq_len", "batch_size", "grad_accum", "lora_r", "model_name"]},
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print GPU info from parent (both GPUs visible)
    import torch
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    all_results = []

    with open(RESULTS_FILE, "w") as f:
        # Phase 1: Baseline runs
        print("\n" + "=" * 70)
        print("PHASE 1: Baseline runs (fixed configs per model)")
        print("=" * 70)

        for cfg in MODEL_CONFIGS:
            print(f"\n{'─' * 60}")
            print(f"Model: {cfg['label']} ({cfg['model_name']})")
            print(f"Config: seq={cfg['seq_len']} bs={cfg['batch_size']} "
                  f"ga={cfg['grad_accum']} r={cfg['lora_r']} "
                  f"gpus={cfg['cuda_devices']}")
            print(f"{'─' * 60}")

            run_cfg = {**cfg, "max_steps": 5}
            result = run_training_subprocess(run_cfg)
            result["phase"] = "baseline"
            all_results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

        # Phase 2: Probe OOM boundaries
        print("\n" + "=" * 70)
        print("PHASE 2: Probing OOM boundaries")
        print("=" * 70)

        for cfg in MODEL_CONFIGS:
            baseline_key = (cfg["seq_len"], cfg["batch_size"])
            oom_hit = False

            for probe_seq, probe_bs in PROBE_CONFIGS:
                if (probe_seq, probe_bs) == baseline_key:
                    continue
                if oom_hit and probe_bs > 1:
                    continue

                print(f"\n  Probe: {cfg['label']} seq={probe_seq} bs={probe_bs}")
                run_cfg = {
                    **cfg,
                    "seq_len": probe_seq,
                    "batch_size": probe_bs,
                    "grad_accum": 1,
                    "max_steps": 3,
                }
                result = run_training_subprocess(run_cfg)
                result["phase"] = "probe"
                all_results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()

                if result["status"] == "OOM" and probe_bs <= 1:
                    oom_hit = True

    # Print summary
    print(f"\n\n{'=' * 80}")
    print("PROGRESSIVE SCALING RESULTS")
    print(f"{'=' * 80}")
    print(f"{'model':>6} {'phase':>8} {'seq':>5} {'bs':>3} {'ga':>3} "
          f"{'status':>6} {'loss':>8} {'steps/s':>7} {'gpu0_gb':>7} {'gpu1_gb':>7} {'time':>6}")
    print("-" * 80)

    for r in all_results:
        loss = f"{r['train_loss']:.4f}" if r.get("train_loss") else ""
        sps = f"{r['train_steps_per_second']:.2f}" if r.get("train_steps_per_second") else ""
        g0 = r.get("gpu0_peak_alloc_gb", "")
        g1 = r.get("gpu1_peak_alloc_gb", "")
        print(f"{r.get('model','?'):>6} {r.get('phase',''):>8} {r.get('seq_len',''):>5} "
              f"{r.get('batch_size',''):>3} {r.get('grad_accum',''):>3} "
              f"{r.get('status','?'):>6} {str(loss):>8} {str(sps):>7} "
              f"{str(g0):>7} {str(g1):>7} {r.get('elapsed_s',''):>6}")

    ok = [r for r in all_results if r["status"] == "OK"]
    oom = [r for r in all_results if r["status"] == "OOM"]
    err = [r for r in all_results if r["status"] not in ("OK", "OOM")]

    print(f"\nTotal: {len(all_results)} runs | OK: {len(ok)} | OOM: {len(oom)} | Error: {len(err)}")
    print(f"Results saved to {RESULTS_FILE}")

    if ok:
        largest_ok = max(ok, key=lambda r: float(r.get("model", "0").rstrip("B")))
        print(f"Largest model trained successfully: {largest_ok['model']} "
              f"(seq={largest_ok['seq_len']}, bs={largest_ok['batch_size']})")


if __name__ == "__main__":
    main()
