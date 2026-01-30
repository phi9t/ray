"""Diagnose: is bnb 4-bit + gradient checkpointing broken on Pascal (CC 6.1)?"""
import gc
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

MODEL = "Qwen/Qwen3-0.6B"
SEQ = 256


def run_test(label, quant_bits, use_gc):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"{'='*50}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    if quant_bits == 4:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    if use_gc:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)

    texts = ["Test text for diagnosis. " * 20] * 16

    def tok_fn(examples):
        out = tok(examples["text"], truncation=True, padding="max_length", max_length=SEQ)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": texts}).map(tok_fn, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="/tmp/diag",
        max_steps=2,
        per_device_train_batch_size=1,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=use_gc,
    )

    try:
        Trainer(model=model, args=args, train_dataset=ds).train()
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  RESULT: OK | peak={peak:.2f} GB")
        return True
    except Exception as e:
        err = str(e)[:150]
        print(f"  RESULT: FAIL | {err}")
        return False
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CC: {torch.cuda.get_device_capability(0)}")
    print(f"bnb version: {__import__('bitsandbytes').__version__}")
    print(f"torch version: {torch.__version__}")

    results = {}
    results["4bit_no_gc"] = run_test("4-bit NF4, NO gradient checkpointing", 4, False)

    # If 4bit without gc fails, CUDA context is corrupted — stop
    if not results["4bit_no_gc"]:
        print("\n4-bit without GC already fails — bnb 4-bit broken on this GPU")
        return

    results["4bit_gc"] = run_test("4-bit NF4, WITH gradient checkpointing", 4, True)

    # If CUDA context got corrupted, we can't continue
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        print("\nCUDA context corrupted after 4bit+gc test")
        return

    results["8bit_gc"] = run_test("8-bit, WITH gradient checkpointing", 8, True)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
