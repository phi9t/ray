"""Push Qwen3-0.6B LoRA to longer sequences using memory optimizations.

Techniques:
  1. Gradient checkpointing — recompute activations during backward pass
  2. Gradient accumulation — simulate larger batch without more VRAM
  3. 8-bit optimizer (bitsandbytes AdamW8bit) — halve optimizer state memory
  4. QLoRA (4-bit quantized base model) — ~4x model memory reduction
  5. Combinations of the above
"""

import gc
import json
import os
import time
import torch

RESULTS_FILE = "/data/outputs/qwen3-scaling/long_seq_results.jsonl"
MODEL_NAME = "Qwen/Qwen3-0.6B"


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_config(cfg, device="cuda:0"):
    cleanup()

    label = (f"seq={cfg['seq_len']:<5} bs={cfg['batch_size']:<2} "
             f"r={cfg['lora_rank']:<3} ga={cfg['grad_accum']:<2} "
             f"gc={'Y' if cfg['grad_ckpt'] else 'N'} "
             f"opt={cfg['optimizer']:<8} "
             f"quant={cfg['quant'] or 'none':<5} "
             f"tgt={cfg['n_targets']}")

    result = {**cfg, "device": str(device)}

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        all_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
        targets = all_targets[:cfg["n_targets"]]

        # Model loading
        model_kwargs = {}
        if cfg["quant"] == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": device}
        elif cfg["quant"] == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = {"": device}
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        if cfg["quant"] in ("4bit", "8bit"):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["grad_ckpt"]
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_rank"] * 2,
            lora_dropout=0.05,
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)

        if cfg["grad_ckpt"] and cfg["quant"] not in ("4bit", "8bit"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if cfg["quant"] not in ("4bit", "8bit"):
            model.to(device)
        model.train()

        trainable, total = model.get_nb_trainable_parameters()
        result["trainable_params"] = trainable
        result["total_params"] = total

        model_mem = torch.cuda.memory_allocated(device) / 1024**2
        result["model_mem_mb"] = round(model_mem, 1)
        total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024**2
        result["gpu_total_mb"] = round(total_gpu, 1)

        # Optimizer
        if cfg["optimizer"] == "adamw8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
        elif cfg["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        seq_len = cfg["seq_len"]
        batch_size = cfg["batch_size"]
        grad_accum = cfg["grad_accum"]

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        torch.cuda.reset_peak_memory_stats(device)
        times = []

        for step in range(3):
            t0 = time.time()
            optimizer.zero_grad()

            for micro in range(grad_accum):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()

            optimizer.step()
            torch.cuda.synchronize(device)
            times.append(time.time() - t0)

        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

        result["status"] = "OK"
        result["peak_alloc_mb"] = round(peak_alloc, 1)
        result["peak_reserved_mb"] = round(peak_reserved, 1)
        result["headroom_mb"] = round(total_gpu - peak_reserved, 1)
        result["avg_step_ms"] = round(1000 * sum(times) / len(times), 1)
        eff_bs = batch_size * grad_accum
        result["effective_batch_size"] = eff_bs
        result["tokens_per_step"] = eff_bs * seq_len
        result["tokens_per_sec"] = round(eff_bs * seq_len / (sum(times) / len(times)), 1)

        print(f"  OK   {label}  peak={peak_alloc:>6.0f}MB "
              f"hdroom={total_gpu - peak_reserved:>5.0f}MB "
              f"{times[-1]*1000:>7.0f}ms "
              f"{result['tokens_per_sec']:>6.0f} tok/s "
              f"eff_bs={eff_bs}")

        del model, optimizer, input_ids, labels, outputs, loss

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        print(f"  OOM  {label}")
        try:
            del model, optimizer
        except:
            pass

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:300]
        print(f"  ERR  {label}  {str(e)[:100]}")
        try:
            del model
        except:
            pass

    cleanup()
    return result


def main():
    os.makedirs("/data/outputs/qwen3-scaling", exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)} — "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    # Check bitsandbytes availability
    try:
        import bitsandbytes as bnb
        has_bnb = True
        print("bitsandbytes: available")
    except ImportError:
        has_bnb = False
        print("bitsandbytes: NOT available (install with: pip install bitsandbytes)")

    all_results = []

    def run(cfg):
        r = test_config(cfg)
        all_results.append(r)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r["status"]

    base = {"lora_rank": 8, "n_targets": 2, "grad_accum": 1,
            "grad_ckpt": False, "optimizer": "adamw", "quant": None}

    # ═══════════════════════════════════════════════════════════════
    # TECHNIQUE 1: Gradient Checkpointing
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TECHNIQUE 1: Gradient Checkpointing (saves activation memory)")
    print("=" * 90)

    print("\n— Baseline (no optimization) —")
    for seq in [512, 1024]:
        run({**base, "seq_len": seq, "batch_size": 1})

    print("\n— With gradient checkpointing —")
    for seq in [512, 1024, 2048, 4096]:
        for bs in [1, 2, 4, 8]:
            status = run({**base, "seq_len": seq, "batch_size": bs, "grad_ckpt": True})
            if status == "OOM":
                break

    # ═══════════════════════════════════════════════════════════════
    # TECHNIQUE 2: Gradient Checkpointing + 8-bit Optimizer
    # ═══════════════════════════════════════════════════════════════
    if has_bnb:
        print("\n" + "=" * 90)
        print("TECHNIQUE 2: Grad Checkpointing + 8-bit AdamW")
        print("=" * 90)

        for seq in [1024, 2048, 4096]:
            for bs in [1, 2, 4, 8]:
                status = run({**base, "seq_len": seq, "batch_size": bs,
                              "grad_ckpt": True, "optimizer": "adamw8bit"})
                if status == "OOM":
                    break

    # ═══════════════════════════════════════════════════════════════
    # TECHNIQUE 3: QLoRA (4-bit quantization)
    # ═══════════════════════════════════════════════════════════════
    if has_bnb:
        print("\n" + "=" * 90)
        print("TECHNIQUE 3: QLoRA (4-bit NF4 quantized base model)")
        print("=" * 90)

        print("\n— QLoRA only —")
        for seq in [512, 1024, 2048, 4096]:
            for bs in [1, 2, 4, 8]:
                status = run({**base, "seq_len": seq, "batch_size": bs, "quant": "4bit"})
                if status == "OOM":
                    break

        print("\n— QLoRA + gradient checkpointing —")
        for seq in [512, 1024, 2048, 4096, 8192]:
            for bs in [1, 2, 4, 8, 16]:
                status = run({**base, "seq_len": seq, "batch_size": bs,
                              "quant": "4bit", "grad_ckpt": True})
                if status == "OOM":
                    break

    # ═══════════════════════════════════════════════════════════════
    # TECHNIQUE 4: QLoRA + GC + 8-bit optimizer + larger LoRA
    # ═══════════════════════════════════════════════════════════════
    if has_bnb:
        print("\n" + "=" * 90)
        print("TECHNIQUE 4: QLoRA + Grad Ckpt + AdamW8bit — max LoRA rank & targets")
        print("=" * 90)

        for rank, n_tgt in [(16, 4), (32, 4), (64, 4), (16, 7), (32, 7)]:
            print(f"\n— r={rank}, {n_tgt} targets —")
            for seq in [1024, 2048, 4096]:
                for bs in [1, 2, 4, 8]:
                    status = run({"seq_len": seq, "batch_size": bs,
                                  "lora_rank": rank, "n_targets": n_tgt,
                                  "grad_accum": 1, "grad_ckpt": True,
                                  "optimizer": "adamw8bit", "quant": "4bit"})
                    if status == "OOM":
                        break

    # ═══════════════════════════════════════════════════════════════
    # TECHNIQUE 5: Gradient accumulation for effective batch scaling
    # ═══════════════════════════════════════════════════════════════
    if has_bnb:
        print("\n" + "=" * 90)
        print("TECHNIQUE 5: Grad accumulation on best long-seq configs")
        print("=" * 90)

        # Find the largest seq that worked at bs=1 with full optimizations
        best_ok = [r for r in all_results
                   if r["status"] == "OK" and r.get("quant") == "4bit"
                   and r.get("grad_ckpt")]
        if best_ok:
            max_seq = max(r["seq_len"] for r in best_ok)
            # Test grad accumulation at that seq length
            for ga in [2, 4, 8, 16]:
                run({"seq_len": max_seq, "batch_size": 1,
                     "lora_rank": 16, "n_targets": 4,
                     "grad_accum": ga, "grad_ckpt": True,
                     "optimizer": "adamw8bit", "quant": "4bit"})

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SUMMARY: Max sequence length achieved per technique")
    print("=" * 90)

    ok = [r for r in all_results if r["status"] == "OK"]

    # Group by technique
    techniques = {}
    for r in ok:
        key_parts = []
        if r.get("quant"):
            key_parts.append(f"QLoRA-{r['quant']}")
        else:
            key_parts.append("fp16")
        if r.get("grad_ckpt"):
            key_parts.append("GradCkpt")
        if r.get("optimizer") == "adamw8bit":
            key_parts.append("Adam8bit")
        key = " + ".join(key_parts) if key_parts else "baseline"
        if key not in techniques:
            techniques[key] = []
        techniques[key].append(r)

    print(f"\n{'Technique':<45} {'MaxSeq':>6} {'MaxBS':>5} "
          f"{'Peak':>7} {'Hdroom':>6} {'tok/s':>7} {'EffBS':>5}")
    print("-" * 90)

    for tech, results in sorted(techniques.items()):
        best_seq = max(results, key=lambda r: r["seq_len"])
        best_at_max = [r for r in results if r["seq_len"] == best_seq["seq_len"]]
        best = max(best_at_max, key=lambda r: r.get("tokens_per_sec", 0))
        print(f"{tech:<45} {best['seq_len']:>6} {best['batch_size']:>5} "
              f"{best.get('peak_alloc_mb', 0):>6.0f}M "
              f"{best.get('headroom_mb', 0):>5.0f}M "
              f"{best.get('tokens_per_sec', 0):>6.0f} "
              f"{best.get('effective_batch_size', best['batch_size']):>5}")

    # Overall best per seq length
    print(f"\n{'Seq':>5} | Best config")
    print("-" * 90)
    for seq in sorted(set(r["seq_len"] for r in ok)):
        at_seq = [r for r in ok if r["seq_len"] == seq]
        best = max(at_seq, key=lambda r: r.get("tokens_per_sec", 0))
        opts = []
        if best.get("quant"):
            opts.append(f"QLoRA-{best['quant']}")
        if best.get("grad_ckpt"):
            opts.append("GradCkpt")
        if best.get("optimizer") == "adamw8bit":
            opts.append("Adam8bit")
        tech = " + ".join(opts) if opts else "baseline"
        print(f"{seq:>5} | bs={best['batch_size']} r={best['lora_rank']} "
              f"tgt={best['n_targets']} ga={best['grad_accum']} "
              f"[{tech}] — "
              f"{best.get('tokens_per_sec', 0):.0f} tok/s, "
              f"peak={best.get('peak_alloc_mb', 0):.0f}MB")

    print(f"\nDetailed results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
