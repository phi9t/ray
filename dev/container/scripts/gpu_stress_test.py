"""Direct GPU stress test for Qwen3-0.6B LoRA — no Ray Train overhead.

Tests increasing batch_size × seq_len × lora_rank until OOM on each GPU.
"""

import gc
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen3-0.6B"
RESULTS_FILE = "/data/outputs/qwen3-scaling/gpu_stress_results.jsonl"


def get_gpu_mem():
    """Return (allocated_MB, reserved_MB, total_MB) for current device."""
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
        torch.cuda.get_device_properties(0).total_memory / 1024**2,
    )


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_config(batch_size, seq_len, lora_rank, lora_targets, device="cuda:0"):
    """Run a few forward+backward passes and return peak memory."""
    cleanup()

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lora_rank": lora_rank,
        "lora_targets": lora_targets,
        "device": str(device),
    }

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=lora_targets,
        )
        model = get_peft_model(model, lora_config)
        model.to(device)
        model.train()

        trainable, total = model.get_nb_trainable_parameters()
        result["trainable_params"] = trainable
        result["total_params"] = total

        alloc_after_model, _, total_memory = get_gpu_mem()
        result["model_mem_mb"] = round(alloc_after_model, 1)
        result["gpu_total_mb"] = round(total_memory, 1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        # Fake input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        # Forward + backward (3 steps)
        times = []
        for step in range(3):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            times.append(time.time() - t0)

        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

        result["status"] = "OK"
        result["peak_alloc_mb"] = round(peak_alloc, 1)
        result["peak_reserved_mb"] = round(peak_reserved, 1)
        result["headroom_mb"] = round(total_memory - peak_reserved, 1)
        result["avg_step_ms"] = round(1000 * sum(times) / len(times), 1)
        result["tokens_per_sec"] = round(batch_size * seq_len / (sum(times) / len(times)), 1)

        # Clean up
        del model, optimizer, input_ids, labels, outputs, loss
        cleanup()

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        # Try to recover
        try:
            del model, optimizer
        except:
            pass
        cleanup()

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]
        cleanup()

    return result


def main():
    import os
    os.makedirs("/data/outputs/qwen3-scaling", exist_ok=True)

    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} — {props.total_memory / 1024**2:.0f} MB")

    all_results = []

    def run_and_log(bs, seq, rank, targets, device="cuda:0"):
        label = f"bs={bs:>2} seq={seq:>5} r={rank:>2} targets={len(targets)} dev={device}"
        r = test_config(bs, seq, rank, targets, device)
        all_results.append(r)

        if r["status"] == "OK":
            print(f"  OK   {label}  peak={r['peak_alloc_mb']:>7.0f}MB "
                  f"headroom={r['headroom_mb']:>6.0f}MB "
                  f"{r['avg_step_ms']:>7.0f}ms/step "
                  f"{r['tokens_per_sec']:>8.0f} tok/s")
        else:
            print(f"  {r['status']:<4} {label}")

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r["status"]

    # ── Phase 1: Batch size sweep at various seq lengths (r=8, qv only) ──
    qv = ["q_proj", "v_proj"]
    print("\n" + "=" * 80)
    print("PHASE 1: Batch size × Seq length (LoRA r=8, q_proj+v_proj)")
    print("=" * 80)

    for seq in [128, 256, 512, 1024, 2048, 4096]:
        print(f"\n--- seq_len = {seq} ---")
        for bs in [1, 2, 4, 8, 16, 32]:
            status = run_and_log(bs, seq, 8, qv)
            if status == "OOM":
                print(f"       OOM at bs={bs}, skipping larger batch sizes")
                break

    # ── Phase 2: LoRA rank sweep ──
    print("\n" + "=" * 80)
    print("PHASE 2: LoRA rank scaling (bs=4, seq=512)")
    print("=" * 80)

    for rank in [8, 16, 32, 64, 128]:
        status = run_and_log(4, 512, rank, qv)
        if status == "OOM":
            break

    # ── Phase 3: More LoRA target modules ──
    all_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    print("\n" + "=" * 80)
    print("PHASE 3: LoRA target module scaling (bs=4, seq=512, r=16)")
    print("=" * 80)

    for n in [2, 4, 7]:
        targets = all_targets[:n]
        status = run_and_log(4, 512, 16, targets)
        if status == "OOM":
            break

    # ── Phase 4: Max throughput — find sweet spot ──
    print("\n" + "=" * 80)
    print("PHASE 4: Max throughput search (r=16, 4 targets)")
    print("=" * 80)

    qkvo = ["q_proj", "k_proj", "v_proj", "o_proj"]
    for seq in [256, 512, 1024]:
        print(f"\n--- seq_len = {seq} ---")
        for bs in [1, 2, 4, 8, 16, 32]:
            status = run_and_log(bs, seq, 16, qkvo)
            if status == "OOM":
                break

    # ── Phase 5: Second GPU ──
    if torch.cuda.device_count() >= 2:
        print("\n" + "=" * 80)
        print("PHASE 5: Second GPU (cuda:1) verification")
        print("=" * 80)
        for bs in [4, 8, 16]:
            status = run_and_log(bs, 512, 16, qkvo, device="cuda:1")
            if status == "OOM":
                break

    # ── Summary ──
    print("\n\n" + "=" * 80)
    print("FULL RESULTS SUMMARY")
    print("=" * 80)
    ok = [r for r in all_results if r["status"] == "OK"]
    oom = [r for r in all_results if r["status"] == "OOM"]

    if ok:
        best_tps = max(ok, key=lambda r: r["tokens_per_sec"])
        best_bs = max(ok, key=lambda r: r["batch_size"] * r["seq_len"])
        most_params = max(ok, key=lambda r: r["trainable_params"])

        print(f"\nTotal configs tested: {len(all_results)} "
              f"({len(ok)} OK, {len(oom)} OOM)")
        print(f"\nBest throughput:")
        print(f"  {best_tps['tokens_per_sec']:.0f} tok/s — "
              f"bs={best_tps['batch_size']} seq={best_tps['seq_len']} "
              f"r={best_tps['lora_rank']} "
              f"peak={best_tps['peak_alloc_mb']:.0f}MB")
        print(f"\nLargest batch×seq:")
        print(f"  bs={best_bs['batch_size']}×seq={best_bs['seq_len']} = "
              f"{best_bs['batch_size']*best_bs['seq_len']} tokens/step — "
              f"r={best_bs['lora_rank']} "
              f"peak={best_bs['peak_alloc_mb']:.0f}MB")
        print(f"\nMost trainable params:")
        print(f"  {most_params['trainable_params']:,} params — "
              f"r={most_params['lora_rank']} "
              f"targets={most_params['lora_targets']}")

        print(f"\nOOM boundaries:")
        for r in oom:
            print(f"  bs={r['batch_size']} seq={r['seq_len']} "
                  f"r={r['lora_rank']} targets={len(r['lora_targets'])}")

    print(f"\nDetailed results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
