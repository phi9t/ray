"""Fine-tune Qwen3-0.6B with Ray Train + PEFT/LoRA."""

import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "/data/outputs/qwen3-training"


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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    # LoRA config for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Simple causal LM training data
    texts = [
        "Ray is a distributed computing framework for Python.",
        "Large language models can be fine-tuned with LoRA adapters.",
        "Distributed training scales across multiple GPUs and nodes.",
        "HuggingFace Transformers provides pretrained model checkpoints.",
        "Parameter-efficient fine-tuning reduces memory requirements.",
    ] * 20  # Repeat for a minimal training set

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=config.get("epochs", 2),
        per_device_train_batch_size=config.get("batch_size", 4),
        learning_rate=config.get("lr", 2e-4),
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[RayTrainReportCallback()],
    )
    trainer = prepare_trainer(trainer)
    trainer.train()


def main():
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"epochs": 2, "batch_size": 2, "lr": 2e-4},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(
            storage_path=OUTPUT_DIR,
            name="qwen3-lora-finetune",
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )
    result = trainer.fit()
    print(f"\nTraining complete. Last checkpoint: {result.checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
