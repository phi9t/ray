#!/usr/bin/env python3
"""Long-term LoRA training for Qwen3 models with best configurations.

Supports three training modes based on experiment results:
- 0.6B: DDP (multi-GPU), seq=256, bs=4, fp16 (2581 tok/s)
- 1.7B: Single-GPU QLoRA+GC, seq=1024, bs=1, ga=8 (391 tok/s)
- 4B: Pipeline parallel, QLoRA+GC, seq=512, bs=2 (186 tok/s)

Features:
- FineWeb-Edu dataset with train/val split
- Periodic evaluation with loss logging
- Checkpoint saving
- TensorBoard + JSONL metrics for live monitoring
- Configurable via command-line args

Hardware: 2x NVIDIA GTX 1070 Ti (8GB each)
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# Best configurations from experiments
MODEL_CONFIGS = {
    "0.6B": {
        "model_name": "Qwen/Qwen3-0.6B",
        "strategy": "ddp",  # Use torchrun --nproc_per_node=2
        "seq_len": 256,
        "batch_size": 4,
        "grad_accum": 1,
        "lora_rank": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "fp16": True,
        "grad_ckpt": False,
        "quant": None,  # DDP doesn't work with bnb quantization
        "device_map": None,
        "expected_tok_s": 2581,
        "expected_vram_mb": 6121,
    },
    "1.7B": {
        "model_name": "Qwen/Qwen3-1.7B",
        "strategy": "single",
        "seq_len": 1024,
        "batch_size": 1,
        "grad_accum": 8,  # Effective batch = 8
        "lora_rank": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "fp16": True,
        "grad_ckpt": True,
        "quant": "4bit",
        "device_map": {"": 0},  # cuda:0 only
        "expected_tok_s": 391,
        "expected_vram_mb": 5714,
    },
    "4B": {
        "model_name": "Qwen/Qwen3-4B",
        "strategy": "pipeline",
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum": 4,  # Effective batch = 8
        "lora_rank": 8,
        "lora_targets": ["q_proj", "v_proj"],
        "fp16": True,
        "grad_ckpt": True,
        "quant": "4bit",
        "device_map": "auto",  # Splits across both GPUs
        "expected_tok_s": 186,
        "expected_vram_mb": 4313,
    },
}


class MetricsLoggerCallback(TrainerCallback):
    """Callback to log metrics to JSONL file for live tailing."""

    def __init__(self, metrics_file, seq_len, expected_tok_s):
        self.metrics_file = metrics_file
        self.seq_len = seq_len
        self.expected_tok_s = expected_tok_s
        self.train_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        timestamp = datetime.now().isoformat()

        # Calculate tokens per second
        tokens_per_sec = None
        if "train_steps_per_second" in logs:
            effective_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps
            n_gpus = max(1, torch.cuda.device_count() if args.local_rank == -1 else 1)
            tokens_per_sec = logs["train_steps_per_second"] * effective_bs * self.seq_len * n_gpus

        # Get peak VRAM
        peak_vram_mb = None
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        metrics = {
            "step": step,
            "timestamp": timestamp,
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
            "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
            "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
            "epoch": logs.get("epoch"),
        }

        # Remove None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if v is not None}

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Also print to stdout
        if "loss" in logs:
            tok_s_str = f"{tokens_per_sec:.0f}" if tokens_per_sec else "?"
            print(f"[Step {step}] loss={logs['loss']:.4f} tok/s={tok_s_str}")
        if "eval_loss" in logs:
            print(f"[Step {step}] eval_loss={logs['eval_loss']:.4f}")


def load_fineweb_dataset(tokenizer, seq_len, num_samples=50000, val_ratio=0.1):
    """Load FineWeb-Edu dataset with train/val split."""
    print(f"Loading FineWeb-Edu dataset ({num_samples} samples)...")

    try:
        # Try streaming mode for FineWeb-Edu
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        # Take a subset
        samples = list(dataset.take(num_samples))
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
    except Exception as e:
        print(f"Could not load FineWeb-Edu: {e}")
        print("Falling back to synthetic dataset...")
        # Fallback to synthetic data
        texts = ["Ray is a unified framework for scaling AI applications. " * 50] * num_samples
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": texts})

    # Tokenize
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Split into train/val
    split = tokenized.train_test_split(test_size=val_ratio, seed=42)
    print(f"Train samples: {len(split['train'])}, Val samples: {len(split['test'])}")
    return split["train"], split["test"]


def setup_model(config):
    """Load and configure model based on config."""
    model_name = config["model_name"]
    quant = config["quant"]
    device_map = config["device_map"]
    grad_ckpt = config["grad_ckpt"]

    print(f"Loading model: {model_name}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Quantization: {quant or 'none'}")
    print(f"  Device map: {device_map}")
    print(f"  Gradient checkpointing: {grad_ckpt}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {"torch_dtype": torch.float16}

    if quant == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if device_map:
        model_kwargs["device_map"] = device_map

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare for k-bit training if quantized
    if quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt
        )

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        lora_dropout=0.05,
        target_modules=config["lora_targets"],
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for non-quantized models
    if grad_ckpt and quant != "4bit":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Print model info
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Log device placement
    devices = set()
    for _, p in model.named_parameters():
        devices.add(str(p.device))
    print(f"  Param devices: {sorted(devices)}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Long-term Qwen3 LoRA training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["0.6B", "1.7B", "4B"],
        help="Model size to train",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Total training steps (default: 10000)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps (default: 100)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: /data/outputs/qwen3-longrun-{model})",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of dataset samples (default: 50000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    args = parser.parse_args()

    # Get config for selected model
    config = MODEL_CONFIGS[args.model]
    model_label = args.model

    # Output directory
    output_dir = args.output_dir or f"/data/outputs/qwen3-longrun-{model_label.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    tensorboard_dir = os.path.join(output_dir, "runs")

    print("=" * 70)
    print(f"LONG-TERM TRAINING: Qwen3-{model_label}")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Metrics file: {metrics_file}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Total steps: {args.num_steps}")
    print(f"Eval every: {args.eval_steps} steps")
    print(f"Save every: {args.save_steps} steps")
    print()

    # Print expected performance
    print(f"Expected throughput: ~{config['expected_tok_s']} tok/s")
    print(f"Expected peak VRAM: ~{config['expected_vram_mb']} MB")
    tokens_per_step = config["batch_size"] * config["grad_accum"] * config["seq_len"]
    est_time_s = args.num_steps * tokens_per_step / config["expected_tok_s"]
    print(f"Estimated training time: {est_time_s / 3600:.1f} hours")
    print()

    # GPU info
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    # Load model and tokenizer
    model, tokenizer = setup_model(config)

    # Load dataset
    train_dataset, eval_dataset = load_fineweb_dataset(
        tokenizer,
        config["seq_len"],
        num_samples=args.num_samples,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.num_steps,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit" if config["quant"] else "adamw_torch",
        fp16=config["fp16"],
        logging_dir=tensorboard_dir,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep last 3 checkpoints
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=config["grad_ckpt"],
        ddp_find_unused_parameters=False if config["strategy"] == "ddp" else None,
        # For DDP, disable auto-detection that breaks with device_map
        ddp_backend="nccl" if config["strategy"] == "ddp" else None,
    )

    # Create trainer
    metrics_callback = MetricsLoggerCallback(
        metrics_file, config["seq_len"], config["expected_tok_s"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[metrics_callback],
    )

    # Write initial metadata
    metadata = {
        "model": model_label,
        "model_name": config["model_name"],
        "strategy": config["strategy"],
        "config": {k: v for k, v in config.items() if k not in ["model_name"]},
        "training_args": {
            "num_steps": args.num_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "learning_rate": args.learning_rate,
            "num_samples": args.num_samples,
        },
        "start_time": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Train
    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    try:
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            trainer.train()

        # Save final model
        final_dir = os.path.join(output_dir, "final")
        print(f"\nSaving final model to {final_dir}")
        trainer.save_model(final_dir)

        # Write completion metadata
        with open(os.path.join(output_dir, "training_metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "completed"
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        with open(os.path.join(output_dir, "training_metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "interrupted"
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        with open(os.path.join(output_dir, "training_metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        raise


if __name__ == "__main__":
    main()
