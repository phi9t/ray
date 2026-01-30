"""Test: bnb 4-bit + HF Trainer DataParallel conflict on multi-GPU.

Bug: Trainer wraps in nn.DataParallel when n_gpu > 1. It skips DP for 8-bit
(checks is_loaded_in_8bit) but NOT 4-bit. DP replicates bnb 4-bit tensors
to other GPUs -> "illegal memory access", corrupts CUDA context.

Fix: Set CUDA_VISIBLE_DEVICES to limit visible GPUs. With 1 visible GPU,
Trainer can't trigger DataParallel. For multi-GPU splits (8B model),
both GPUs are visible but device_map="auto" distributes layers properly.

Tests run as subprocesses to isolate CUDA context corruption.
"""
import os
import json
import subprocess
import sys
import textwrap

WORKER = textwrap.dedent(r'''
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

MODEL = "Qwen/Qwen3-0.6B"
SEQ = 256
n_gpu = torch.cuda.device_count()
device_map = "auto" if n_gpu > 1 else {"": 0}

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                              device_map=device_map, torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                          lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))

texts = ["Test text. " * 20] * 16
def tok_fn(ex):
    out = tok(ex["text"], truncation=True, padding="max_length", max_length=SEQ)
    out["labels"] = out["input_ids"].copy()
    return out
ds = Dataset.from_dict({"text": texts}).map(tok_fn, batched=True, remove_columns=["text"])
args = TrainingArguments(output_dir="/tmp/dp-test", max_steps=2,
                         per_device_train_batch_size=1, fp16=True,
                         logging_steps=1, save_strategy="no", report_to="none",
                         remove_unused_columns=False, dataloader_pin_memory=False,
                         optim="paged_adamw_8bit", gradient_checkpointing=True)
Trainer(model=model, args=args, train_dataset=ds).train()
peak = torch.cuda.max_memory_allocated(0) / 1024**3
print(json.dumps({"status": "OK", "n_gpu": n_gpu, "peak_gb": round(peak, 2)}))
''')


def run_test(label, cuda_visible):
    print(f"\n=== {label} ===")
    print(f"  CUDA_VISIBLE_DEVICES={cuda_visible}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", WORKER],
            env=env, capture_output=True, text=True, timeout=300,
        )
        for line in proc.stdout.strip().split("\n"):
            try:
                data = json.loads(line)
                if data.get("status") == "OK":
                    print(f"  SUCCESS: n_gpu={data['n_gpu']}, peak={data['peak_gb']}GB")
                    return True
            except json.JSONDecodeError:
                pass
        # Extract error
        for line in proc.stderr.split("\n"):
            if "RuntimeError" in line:
                print(f"  FAIL: {line.strip()[:120]}")
                return False
        print(f"  FAIL: exit code {proc.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print("  FAIL: timeout")
        return False


def main():
    import torch
    n_gpu = torch.cuda.device_count()
    print(f"System GPUs: {n_gpu}")

    # Test 1: Both GPUs visible -> Trainer wraps in DP -> crash
    r1 = run_test("Both GPUs visible (expect FAIL from DP)", "0,1")

    # Test 2: Single GPU visible -> no DP -> works
    r2 = run_test("Single GPU visible (expect SUCCESS)", "0")

    print(f"\n{'='*60}")
    if not r1 and r2:
        print("CONFIRMED: bnb 4-bit + DataParallel = broken")
        print("FIX: CUDA_VISIBLE_DEVICES=0 prevents DP wrapping")
        print("ALL TESTS PASSED")
    elif r1 and r2:
        print("Both passed (DP may work on this GPU). ALL TESTS PASSED")
    else:
        print(f"UNEXPECTED: 2gpu={'OK' if r1 else 'FAIL'}, 1gpu={'OK' if r2 else 'FAIL'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
