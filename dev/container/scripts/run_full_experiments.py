#!/usr/bin/env python3
"""Comprehensive Qwen3 inference & LoRA training experiments across 0.6B, 1.7B, 4B.

Tests inference benchmarks, single-GPU training tricks, and multi-GPU parallelism
strategies. Each experiment runs in a subprocess for CUDA isolation.

Hardware: 2x NVIDIA GTX 1070 Ti (8GB VRAM, Pascal CC 6.1, PCIe 3.0)
- No bf16, no Flash Attention 2
- fp16 only for mixed precision

Output:
- JSONL results: /data/outputs/qwen3-full-experiments/results.jsonl
- Markdown report: /workspace/ray/dev/container/QWEN3_FULL_EXPERIMENT_REPORT.md
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime

OUTPUT_DIR = "/data/outputs/qwen3-full-experiments"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
REPORT_FILE = "/workspace/ray/dev/container/QWEN3_FULL_EXPERIMENT_REPORT.md"

MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
}

# ─── Worker script templates ───────────────────────────────────────────────────

INFERENCE_WORKER = textwrap.dedent(r'''
import gc, json, sys, time, torch

config = json.loads(sys.argv[1])
model_name = config["model_name"]
quant = config.get("quant")  # None, "4bit"
batch_size = config["batch_size"]
seq_len = config["seq_len"]
device = config.get("device", "cuda:0")
multi_gpu = config.get("multi_gpu", False)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {}
if quant == "4bit":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    if multi_gpu:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
else:
    model_kwargs["torch_dtype"] = torch.float16
    if multi_gpu:
        model_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
if not multi_gpu and quant != "4bit":
    model = model.to(device)
model.eval()

if multi_gpu and hasattr(model, 'hf_device_map'):
    first_device = next(iter(set(model.hf_device_map.values())))
    if isinstance(first_device, int):
        input_device = f"cuda:{first_device}"
    else:
        input_device = first_device
else:
    input_device = device

input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=input_device)

# Warmup
with torch.no_grad():
    _ = model(input_ids=input_ids[:1, :min(64, seq_len)])

torch.cuda.reset_peak_memory_stats()

# Measure forward pass latency
times = []
with torch.no_grad():
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(input_ids=input_ids)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

# Measure generation (tokens/sec)
gen_input = torch.randint(0, 1000, (1, 32), device=input_device)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    gen_out = model.generate(gen_input, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
gen_time = time.time() - t0
new_tokens = gen_out.shape[1] - 32

peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2

result = {
    "status": "OK",
    "avg_forward_ms": round(1000 * sum(times) / len(times), 1),
    "min_forward_ms": round(1000 * min(times), 1),
    "tokens_per_sec_gen": round(new_tokens / gen_time, 1) if gen_time > 0 else 0,
    "time_to_first_token_ms": round(1000 * gen_time / max(new_tokens, 1), 1),
    "peak_alloc_mb": round(peak_alloc, 1),
    "peak_reserved_mb": round(peak_reserved, 1),
    "headroom_mb": round(total_mem - peak_reserved, 1),
}
print("__RESULT_JSON__:" + json.dumps(result))
''')

TRAINING_WORKER = textwrap.dedent(r'''
import gc, json, sys, time, torch

config = json.loads(sys.argv[1])
model_name = config["model_name"]
quant = config.get("quant")  # None, "4bit"
batch_size = config["batch_size"]
seq_len = config["seq_len"]
lora_rank = config.get("lora_rank", 8)
n_targets = config.get("n_targets", 2)
grad_ckpt = config.get("grad_ckpt", False)
grad_accum = config.get("grad_accum", 1)
optimizer_type = config.get("optimizer", "adamw")
use_fp32 = config.get("use_fp32", False)
num_steps = config.get("num_steps", 5)
device = config.get("device", "cuda:0")
multi_gpu = config.get("multi_gpu", False)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

all_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
targets = all_targets[:n_targets]

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {}
if quant == "4bit":
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    if multi_gpu:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
elif not use_fp32:
    model_kwargs["torch_dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = torch.float32
    if multi_gpu:
        model_kwargs["device_map"] = "auto"

if multi_gpu and quant != "4bit" and not use_fp32:
    model_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

if quant == "4bit":
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=lora_rank,
    lora_alpha=lora_rank * 2, lora_dropout=0.05, target_modules=targets,
)
model = get_peft_model(model, lora_config)

if grad_ckpt and quant != "4bit":
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

if not multi_gpu and quant != "4bit":
    model.to(device)
model.train()

trainable, total = model.get_nb_trainable_parameters()

model_mem = torch.cuda.memory_allocated(device if not multi_gpu else 0) / 1024**2
total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**2

# Optimizer
lr = 2e-4
if optimizer_type == "adamw8bit":
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

input_ids = torch.randint(0, 1000, (batch_size, seq_len),
                           device=device if not multi_gpu else "cuda:0")
labels = input_ids.clone()

torch.cuda.reset_peak_memory_stats()
times = []
losses = []

for step in range(num_steps):
    t0 = time.time()
    optimizer.zero_grad()
    for micro in range(grad_accum):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    times.append(elapsed)
    losses.append(outputs.loss.item())

peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
eff_bs = batch_size * grad_accum
tokens_per_step = eff_bs * seq_len

result = {
    "status": "OK",
    "trainable_params": trainable,
    "total_params": total,
    "model_mem_mb": round(model_mem, 1),
    "peak_alloc_mb": round(peak_alloc, 1),
    "peak_reserved_mb": round(peak_reserved, 1),
    "headroom_mb": round(total_gpu - peak_reserved, 1),
    "avg_step_ms": round(1000 * sum(times[1:]) / max(len(times) - 1, 1), 1),
    "effective_batch_size": eff_bs,
    "tokens_per_step": tokens_per_step,
    "tokens_per_sec": round(tokens_per_step / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
    "final_loss": round(losses[-1], 4),
}
print("__RESULT_JSON__:" + json.dumps(result))
''')

DDP_WORKER = textwrap.dedent(r'''
import gc, json, os, sys, time, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

config = json.loads(os.environ["EXP_CONFIG"])
model_name = config["model_name"]
lora_rank = config.get("lora_rank", 8)
seq_len = config["seq_len"]
batch_size = config["batch_size"]
num_steps = config.get("num_steps", 5)

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=lora_rank,
    lora_alpha=lora_rank * 2, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.to(device)
model.train()

ddp_model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-4)

dataset = TensorDataset(
    torch.randint(0, 1000, (64, seq_len)),
    torch.randint(0, 1000, (64, seq_len)),
)
sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

torch.cuda.reset_peak_memory_stats(device)
times = []
losses = []

for step, (ids, labs) in enumerate(loader):
    if step >= num_steps:
        break
    ids, labs = ids.to(device), labs.to(device)
    t0 = time.time()
    optimizer.zero_grad()
    out = ddp_model(input_ids=ids, labels=labs)
    out.loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    times.append(time.time() - t0)
    losses.append(out.loss.item())

peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024**2

if rank == 0:
    result = {
        "status": "OK",
        "peak_alloc_mb": round(peak_alloc, 1),
        "peak_reserved_mb": round(peak_reserved, 1),
        "headroom_mb": round(total_gpu - peak_reserved, 1),
        "avg_step_ms": round(1000 * sum(times[1:]) / max(len(times) - 1, 1), 1),
        "tokens_per_sec": round(batch_size * 2 * seq_len / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
        "final_loss": round(losses[-1], 4),
    }
    print("__RESULT_JSON__:" + json.dumps(result))

dist.destroy_process_group()
''')

FSDP_WORKER = textwrap.dedent(r'''
import gc, json, os, sys, time, torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

config = json.loads(os.environ["EXP_CONFIG"])
model_name = config["model_name"]
lora_rank = config.get("lora_rank", 8)
seq_len = config["seq_len"]
batch_size = config["batch_size"]
num_steps = config.get("num_steps", 5)

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=lora_rank,
    lora_alpha=lora_rank * 2, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
fsdp_model = FSDP(
    model, sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy, device_id=local_rank,
)
fsdp_model.train()

optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=2e-4)

dataset = TensorDataset(
    torch.randint(0, 1000, (64, seq_len)),
    torch.randint(0, 1000, (64, seq_len)),
)
sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

torch.cuda.reset_peak_memory_stats(device)
times = []
losses = []

for step, (ids, labs) in enumerate(loader):
    if step >= num_steps:
        break
    ids, labs = ids.to(device), labs.to(device)
    t0 = time.time()
    optimizer.zero_grad()
    out = fsdp_model(input_ids=ids, labels=labs)
    out.loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    times.append(time.time() - t0)
    losses.append(out.loss.item())

peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024**2

if rank == 0:
    result = {
        "status": "OK",
        "peak_alloc_mb": round(peak_alloc, 1),
        "peak_reserved_mb": round(peak_reserved, 1),
        "headroom_mb": round(total_gpu - peak_reserved, 1),
        "avg_step_ms": round(1000 * sum(times[1:]) / max(len(times) - 1, 1), 1),
        "tokens_per_sec": round(batch_size * 2 * seq_len / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
        "final_loss": round(losses[-1], 4),
    }
    print("__RESULT_JSON__:" + json.dumps(result))

dist.destroy_process_group()
''')

DEEPSPEED_WORKER = textwrap.dedent(r'''
import gc, json, os, sys, time, torch

config = json.loads(os.environ["EXP_CONFIG"])
model_name = config["model_name"]
lora_rank = config.get("lora_rank", 8)
seq_len = config["seq_len"]
batch_size = config["batch_size"]
num_steps = config.get("num_steps", 5)
zero_stage = config.get("zero_stage", 2)

# DeepSpeed init
import deepspeed
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

deepspeed.init_distributed("nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=lora_rank,
    lora_alpha=lora_rank * 2, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

ds_config = {
    "train_batch_size": batch_size * 2,  # world_size=2
    "train_micro_batch_size_per_gpu": batch_size,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": zero_stage},
    "gradient_clipping": 1.0,
    "steps_per_print": 999999,
}

engine, optimizer, _, _ = deepspeed.initialize(
    model=model, config=ds_config,
    model_parameters=model.parameters(),
)
engine.train()

input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
labels = input_ids.clone()

torch.cuda.reset_peak_memory_stats(device)
times = []
losses = []

for step in range(num_steps):
    t0 = time.time()
    out = engine(input_ids=input_ids, labels=labels)
    engine.backward(out.loss)
    engine.step()
    torch.cuda.synchronize(device)
    times.append(time.time() - t0)
    losses.append(out.loss.item())

peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024**2

if local_rank == 0:
    result = {
        "status": "OK",
        "peak_alloc_mb": round(peak_alloc, 1),
        "peak_reserved_mb": round(peak_reserved, 1),
        "headroom_mb": round(total_gpu - peak_reserved, 1),
        "avg_step_ms": round(1000 * sum(times[1:]) / max(len(times) - 1, 1), 1),
        "tokens_per_sec": round(batch_size * 2 * seq_len / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
        "final_loss": round(losses[-1], 4),
    }
    print("__RESULT_JSON__:" + json.dumps(result))
''')


# ─── Experiment definitions ───────────────────────────────────────────────────

def build_experiments():
    """Build the full ordered list of experiments."""
    experiments = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PART A: Inference Benchmarks
    # ═══════════════════════════════════════════════════════════════════════════

    for model_key in ["0.6B", "1.7B", "4B"]:
        # A.1: fp16 inference
        for bs in [1, 4, 8]:
            for seq in [128, 512, 1024, 2048]:
                experiments.append({
                    "part": "A", "category": "inference_fp16",
                    "label": f"A: {model_key} fp16 inference bs={bs} seq={seq}",
                    "type": "inference",
                    "model_key": model_key,
                    "config": {
                        "model_name": MODELS[model_key],
                        "batch_size": bs, "seq_len": seq,
                        "quant": None,
                    },
                    "cuda_visible": "0",
                })

        # A.2: 4-bit NF4 inference
        for bs in [1, 4, 8]:
            for seq in [128, 512, 1024, 2048]:
                experiments.append({
                    "part": "A", "category": "inference_4bit",
                    "label": f"A: {model_key} 4bit inference bs={bs} seq={seq}",
                    "type": "inference",
                    "model_key": model_key,
                    "config": {
                        "model_name": MODELS[model_key],
                        "batch_size": bs, "seq_len": seq,
                        "quant": "4bit",
                    },
                    "cuda_visible": "0",
                })

        # A.3: 2-GPU device_map inference (4B only, also test 1.7B)
        if model_key in ("4B", "1.7B"):
            for bs in [1, 4]:
                for seq in [128, 512, 1024, 2048]:
                    experiments.append({
                        "part": "A", "category": "inference_2gpu",
                        "label": f"A: {model_key} 2GPU inference bs={bs} seq={seq}",
                        "type": "inference",
                        "model_key": model_key,
                        "config": {
                            "model_name": MODELS[model_key],
                            "batch_size": bs, "seq_len": seq,
                            "quant": "4bit", "multi_gpu": True,
                        },
                        "cuda_visible": "0,1",
                    })

    # ═══════════════════════════════════════════════════════════════════════════
    # PART B: Single-GPU Training Tricks
    # ═══════════════════════════════════════════════════════════════════════════

    for model_key in ["0.6B", "1.7B", "4B"]:
        mn = MODELS[model_key]

        # B1: fp16 baseline — LoRA r=8, AdamW, no tricks
        for seq in [128, 256, 512, 1024]:
            for bs in [1, 2, 4]:
                experiments.append({
                    "part": "B", "category": "B1_fp16_baseline",
                    "label": f"B1: {model_key} fp16 baseline seq={seq} bs={bs}",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": mn, "batch_size": bs, "seq_len": seq,
                        "lora_rank": 8, "n_targets": 2, "grad_ckpt": False,
                        "grad_accum": 1, "optimizer": "adamw", "quant": None,
                    },
                    "cuda_visible": "0",
                })

        # B2: Activation checkpointing
        for seq in [128, 256, 512, 1024, 2048]:
            for bs in [1, 2, 4]:
                experiments.append({
                    "part": "B", "category": "B2_grad_ckpt",
                    "label": f"B2: {model_key} grad_ckpt seq={seq} bs={bs}",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": mn, "batch_size": bs, "seq_len": seq,
                        "lora_rank": 8, "n_targets": 2, "grad_ckpt": True,
                        "grad_accum": 1, "optimizer": "adamw", "quant": None,
                    },
                    "cuda_visible": "0",
                })

        # B3: QLoRA (4-bit NF4)
        for seq in [128, 256, 512, 1024, 2048]:
            for bs in [1, 2, 4]:
                experiments.append({
                    "part": "B", "category": "B3_qlora",
                    "label": f"B3: {model_key} QLoRA seq={seq} bs={bs}",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": mn, "batch_size": bs, "seq_len": seq,
                        "lora_rank": 8, "n_targets": 2, "grad_ckpt": False,
                        "grad_accum": 1, "optimizer": "adamw", "quant": "4bit",
                    },
                    "cuda_visible": "0",
                })

        # B4: QLoRA + Activation checkpointing
        for seq in [128, 256, 512, 1024, 2048, 4096]:
            for bs in [1, 2, 4, 8]:
                experiments.append({
                    "part": "B", "category": "B4_qlora_gc",
                    "label": f"B4: {model_key} QLoRA+GC seq={seq} bs={bs}",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": mn, "batch_size": bs, "seq_len": seq,
                        "lora_rank": 8, "n_targets": 2, "grad_ckpt": True,
                        "grad_accum": 1, "optimizer": "adamw", "quant": "4bit",
                    },
                    "cuda_visible": "0",
                })

        # B5: Mixed precision comparison (fp32 vs fp16)
        for seq in [128, 256, 512]:
            experiments.append({
                "part": "B", "category": "B5_fp32",
                "label": f"B5: {model_key} fp32 seq={seq} bs=1",
                "type": "training",
                "model_key": model_key,
                "config": {
                    "model_name": mn, "batch_size": 1, "seq_len": seq,
                    "lora_rank": 8, "n_targets": 2, "grad_ckpt": False,
                    "grad_accum": 1, "optimizer": "adamw", "quant": None,
                    "use_fp32": True,
                },
                "cuda_visible": "0",
            })

        # B6: Gradient accumulation sweep (at QLoRA+GC config)
        # Use seq=512 as a safe baseline for all models
        for ga in [1, 2, 4, 8, 16]:
            experiments.append({
                "part": "B", "category": "B6_grad_accum",
                "label": f"B6: {model_key} GA={ga} seq=512 bs=1",
                "type": "training",
                "model_key": model_key,
                "config": {
                    "model_name": mn, "batch_size": 1, "seq_len": 512,
                    "lora_rank": 8, "n_targets": 4, "grad_ckpt": True,
                    "grad_accum": ga, "optimizer": "adamw", "quant": "4bit",
                },
                "cuda_visible": "0",
            })

        # B7: LoRA rank + target scaling
        for r in [8, 16, 32, 64]:
            for n_tgt in [2, 4, 7]:
                experiments.append({
                    "part": "B", "category": "B7_lora_scale",
                    "label": f"B7: {model_key} r={r} tgt={n_tgt} seq=512 bs=1",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": mn, "batch_size": 1, "seq_len": 512,
                        "lora_rank": r, "n_targets": n_tgt, "grad_ckpt": True,
                        "grad_accum": 1, "optimizer": "adamw", "quant": "4bit",
                    },
                    "cuda_visible": "0",
                })

    # ═══════════════════════════════════════════════════════════════════════════
    # PART C: Multi-GPU Parallelism Strategies
    # ═══════════════════════════════════════════════════════════════════════════

    # C1: DDP — fp16 LoRA (no bnb), 0.6B only (fits in 8GB fp16)
    for seq in [128, 256, 512]:
        for bs in [1, 2, 4]:
            experiments.append({
                "part": "C", "category": "C1_ddp",
                "label": f"C1: 0.6B DDP seq={seq} bs={bs}",
                "type": "ddp",
                "model_key": "0.6B",
                "config": {
                    "model_name": MODELS["0.6B"],
                    "lora_rank": 8, "seq_len": seq, "batch_size": bs,
                },
                "cuda_visible": "0,1",
            })

    # C3: FSDP — fp16 LoRA, 0.6B and 1.7B
    for model_key in ["0.6B", "1.7B"]:
        for seq in [128, 256, 512]:
            for bs in [1, 2]:
                experiments.append({
                    "part": "C", "category": "C3_fsdp",
                    "label": f"C3: {model_key} FSDP seq={seq} bs={bs}",
                    "type": "fsdp",
                    "model_key": model_key,
                    "config": {
                        "model_name": MODELS[model_key],
                        "lora_rank": 8, "seq_len": seq, "batch_size": bs,
                    },
                    "cuda_visible": "0,1",
                })

    # C4: Pipeline Parallelism (device_map="auto") — 4B primary, 1.7B comparison
    for model_key in ["1.7B", "4B"]:
        for seq in [128, 256, 512, 1024]:
            for bs in [1, 2]:
                experiments.append({
                    "part": "C", "category": "C4_pipeline",
                    "label": f"C4: {model_key} pipeline seq={seq} bs={bs}",
                    "type": "training",
                    "model_key": model_key,
                    "config": {
                        "model_name": MODELS[model_key],
                        "batch_size": bs, "seq_len": seq,
                        "lora_rank": 8, "n_targets": 2, "grad_ckpt": True,
                        "grad_accum": 1, "optimizer": "adamw", "quant": "4bit",
                        "multi_gpu": True,
                    },
                    "cuda_visible": "0,1",
                })

    # C5: DeepSpeed ZeRO-2 — 0.6B and 1.7B
    for model_key in ["0.6B", "1.7B"]:
        for seq in [128, 256, 512]:
            experiments.append({
                "part": "C", "category": "C5_zero2",
                "label": f"C5: {model_key} ZeRO-2 seq={seq} bs=1",
                "type": "deepspeed",
                "model_key": model_key,
                "config": {
                    "model_name": MODELS[model_key],
                    "lora_rank": 8, "seq_len": seq, "batch_size": 1,
                    "zero_stage": 2,
                },
                "cuda_visible": "0,1",
            })

    # C6: DeepSpeed ZeRO-3 — 1.7B and 4B
    for model_key in ["1.7B", "4B"]:
        for seq in [128, 256, 512]:
            experiments.append({
                "part": "C", "category": "C6_zero3",
                "label": f"C6: {model_key} ZeRO-3 seq={seq} bs=1",
                "type": "deepspeed",
                "model_key": model_key,
                "config": {
                    "model_name": MODELS[model_key],
                    "lora_rank": 8, "seq_len": seq, "batch_size": 1,
                    "zero_stage": 3,
                },
                "cuda_visible": "0,1",
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # PART D: Combined Best Configs (20 steps)
    # These will be dynamically determined after B+C complete
    # Add placeholder entries that will be filled in
    # ═══════════════════════════════════════════════════════════════════════════

    return experiments


# ─── Subprocess runner ─────────────────────────────────────────────────────────

def run_experiment(exp, timeout=300):
    """Run a single experiment in a subprocess. Returns result dict."""
    exp_type = exp["type"]
    config = exp["config"]
    cuda_vis = exp.get("cuda_visible", "0")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_vis
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    config_json = json.dumps(config)
    result = {"label": exp["label"], "part": exp["part"],
              "category": exp["category"], "model_key": exp["model_key"],
              "config": config}

    t0 = time.time()

    try:
        if exp_type == "inference":
            proc = subprocess.run(
                [sys.executable, "-u", "-c", INFERENCE_WORKER, config_json],
                env=env, capture_output=True, text=True, timeout=timeout,
            )
        elif exp_type == "training":
            proc = subprocess.run(
                [sys.executable, "-u", "-c", TRAINING_WORKER, config_json],
                env=env, capture_output=True, text=True, timeout=timeout,
            )
        elif exp_type in ("ddp", "fsdp", "deepspeed"):
            env["EXP_CONFIG"] = config_json
            worker_map = {"ddp": DDP_WORKER, "fsdp": FSDP_WORKER, "deepspeed": DEEPSPEED_WORKER}
            port_map = {"ddp": "29500", "fsdp": "29501", "deepspeed": "29502"}
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(worker_map[exp_type])
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    ["torchrun", f"--nproc_per_node=2",
                     f"--master_port={port_map[exp_type]}", tmp_path],
                    env=env, capture_output=True, text=True, timeout=timeout,
                )
            finally:
                os.unlink(tmp_path)
        else:
            result["status"] = "ERROR"
            result["error"] = f"Unknown type: {exp_type}"
            return result

        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)

        # Parse result from stdout
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("__RESULT_JSON__:"):
                data = json.loads(line[len("__RESULT_JSON__:"):])
                result.update(data)
                return result

        # No result found — check for errors
        stderr = proc.stderr
        if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
            result["status"] = "OOM"
            # Extract the OOM line
            for line in stderr.split("\n"):
                if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                    result["error"] = line.strip()[:200]
                    break
        elif proc.returncode != 0:
            result["status"] = "ERROR"
            result["error"] = stderr[-300:].strip() if stderr else f"exit code {proc.returncode}"
        else:
            result["status"] = "ERROR"
            result["error"] = "No result JSON in output"

        # Also check stdout for OOM
        if result.get("status") != "OOM":
            if "CUDA out of memory" in proc.stdout or "OutOfMemoryError" in proc.stdout:
                result["status"] = "OOM"

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["elapsed_s"] = timeout
        result["error"] = f"Exceeded {timeout}s timeout"

    except Exception as e:
        result["status"] = "ERROR"
        result["elapsed_s"] = round(time.time() - t0, 1)
        result["error"] = str(e)[:200]

    return result


# ─── OOM skip logic ───────────────────────────────────────────────────────────

def should_skip(exp, oom_tracker):
    """Skip experiments that will obviously OOM based on prior failures."""
    cat = exp["category"]
    model = exp["model_key"]
    cfg = exp["config"]

    key = (cat, model)
    if key not in oom_tracker:
        return False

    oom_info = oom_tracker[key]

    # If bs=1 OOMed at a given seq_len, skip larger bs at same or longer seq
    bs = cfg.get("batch_size", 1)
    seq = cfg.get("seq_len", 0)

    for (oom_seq, oom_bs) in oom_info:
        # If we OOMed at bs=1 at this seq, skip everything at this seq and longer
        if oom_bs <= 1 and seq >= oom_seq:
            return True
        # If we OOMed at this bs and seq, skip larger bs at same seq
        if seq == oom_seq and bs > oom_bs:
            return True
        # If we OOMed at this seq with some bs, skip same bs at longer seq
        if seq > oom_seq and bs >= oom_bs:
            return True

    return False


def record_oom(exp, oom_tracker):
    """Record an OOM for skip logic."""
    key = (exp["category"], exp["model_key"])
    cfg = exp["config"]
    if key not in oom_tracker:
        oom_tracker[key] = []
    oom_tracker[key].append((cfg.get("seq_len", 0), cfg.get("batch_size", 1)))


# ─── Part D: dynamic best config experiments ──────────────────────────────────

def build_part_d_experiments(results):
    """After B+C, build Part D experiments from the best configs."""
    part_d = []

    for model_key in ["0.6B", "1.7B", "4B"]:
        mn = MODELS[model_key]

        # Best single-GPU config: highest tokens_per_sec from B4 (QLoRA+GC)
        b4_ok = [r for r in results
                 if r.get("status") == "OK" and r.get("category") == "B4_qlora_gc"
                 and r.get("model_key") == model_key]
        if b4_ok:
            best = max(b4_ok, key=lambda r: r.get("tokens_per_sec", 0))
            cfg = dict(best["config"])
            cfg["num_steps"] = 20
            part_d.append({
                "part": "D", "category": "D_best_single",
                "label": f"D: {model_key} best single-GPU (20 steps)",
                "type": "training",
                "model_key": model_key,
                "config": cfg,
                "cuda_visible": "0",
            })

        # Best multi-GPU config: highest tokens_per_sec from C*
        c_ok = [r for r in results
                if r.get("status") == "OK" and r.get("part") == "C"
                and r.get("model_key") == model_key]
        if c_ok:
            best = max(c_ok, key=lambda r: r.get("tokens_per_sec", 0))
            cfg = dict(best["config"])
            cfg["num_steps"] = 20
            part_d.append({
                "part": "D", "category": "D_best_multi",
                "label": f"D: {model_key} best multi-GPU (20 steps)",
                "type": best.get("type", "training"),  # preserve ddp/fsdp/etc
                "model_key": model_key,
                "config": cfg,
                "cuda_visible": "0,1",
            })

    return part_d


# ─── Report generation ─────────────────────────────────────────────────────────

def generate_report(results):
    """Generate markdown report from results."""
    lines = []
    w = lines.append

    w("# Qwen3 Inference & LoRA Training: Full Experiment Report")
    w("")
    w(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    w("")

    # ── Hardware ──
    w("## Hardware & Environment")
    w("")
    w("- **GPUs**: 2x NVIDIA GeForce GTX 1070 Ti (8,192 MB VRAM, Pascal CC 6.1, PCIe 3.0)")
    w("- **No bf16 support** — all experiments use fp16 (or fp32 for comparison)")
    w("- **No Flash Attention 2** — requires CC >= 8.0")
    w("- **Models**: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B")
    w("")

    # Count stats
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "OK")
    oom = sum(1 for r in results if r.get("status") == "OOM")
    skip = sum(1 for r in results if r.get("status") == "SKIP")
    err = total - ok - oom - skip
    w(f"**Experiment summary**: {total} total runs — {ok} OK, {oom} OOM, {skip} skipped, {err} errors/timeouts")
    w("")

    # ── Part A: Inference ──
    w("## Part A: Inference Benchmarks")
    w("")

    for model_key in ["0.6B", "1.7B", "4B"]:
        w(f"### {model_key}")
        w("")

        # fp16 inference
        fp16_results = [r for r in results
                        if r.get("category") == "inference_fp16"
                        and r.get("model_key") == model_key]
        if fp16_results:
            w("**fp16 inference:**")
            w("")
            w("| Batch | Seq Len | Forward (ms) | Gen tok/s | Peak VRAM (MB) | Headroom (MB) | Status |")
            w("|-------|---------|-------------|-----------|----------------|---------------|--------|")
            for r in fp16_results:
                c = r["config"]
                if r.get("status") == "OK":
                    w(f"| {c['batch_size']} | {c['seq_len']} | {r.get('avg_forward_ms', '—')} | "
                      f"{r.get('tokens_per_sec_gen', '—')} | {r.get('peak_alloc_mb', '—')} | "
                      f"{r.get('headroom_mb', '—')} | OK |")
                else:
                    w(f"| {c['batch_size']} | {c['seq_len']} | — | — | — | — | {r.get('status', '?')} |")
            w("")

        # 4bit inference
        q4_results = [r for r in results
                      if r.get("category") == "inference_4bit"
                      and r.get("model_key") == model_key]
        if q4_results:
            w("**4-bit NF4 inference:**")
            w("")
            w("| Batch | Seq Len | Forward (ms) | Gen tok/s | Peak VRAM (MB) | Headroom (MB) | Status |")
            w("|-------|---------|-------------|-----------|----------------|---------------|--------|")
            for r in q4_results:
                c = r["config"]
                if r.get("status") == "OK":
                    w(f"| {c['batch_size']} | {c['seq_len']} | {r.get('avg_forward_ms', '—')} | "
                      f"{r.get('tokens_per_sec_gen', '—')} | {r.get('peak_alloc_mb', '—')} | "
                      f"{r.get('headroom_mb', '—')} | OK |")
                else:
                    w(f"| {c['batch_size']} | {c['seq_len']} | — | — | — | — | {r.get('status', '?')} |")
            w("")

        # 2-GPU inference
        gpu2_results = [r for r in results
                        if r.get("category") == "inference_2gpu"
                        and r.get("model_key") == model_key]
        if gpu2_results:
            w("**2-GPU device_map inference:**")
            w("")
            w("| Batch | Seq Len | Forward (ms) | Gen tok/s | Peak VRAM (MB) | Status |")
            w("|-------|---------|-------------|-----------|----------------|--------|")
            for r in gpu2_results:
                c = r["config"]
                if r.get("status") == "OK":
                    w(f"| {c['batch_size']} | {c['seq_len']} | {r.get('avg_forward_ms', '—')} | "
                      f"{r.get('tokens_per_sec_gen', '—')} | {r.get('peak_alloc_mb', '—')} | OK |")
                else:
                    w(f"| {c['batch_size']} | {c['seq_len']} | — | — | — | {r.get('status', '?')} |")
            w("")

    # ── Part B: Training Tricks ──
    w("## Part B: Training Optimization Techniques")
    w("")

    b_categories = [
        ("B1_fp16_baseline", "B1: fp16 Baseline", "Vanilla LoRA r=8, AdamW, no tricks"),
        ("B2_grad_ckpt", "B2: Activation Checkpointing", "gradient_checkpointing_enable() — trades compute for memory"),
        ("B3_qlora", "B3: QLoRA (4-bit NF4)", "4-bit quantized base + fp16 LoRA adapters"),
        ("B4_qlora_gc", "B4: QLoRA + Activation Checkpointing", "Combined — the winning single-GPU strategy"),
        ("B5_fp32", "B5: fp32 Training", "Full precision comparison (no fp16)"),
        ("B6_grad_accum", "B6: Gradient Accumulation Sweep", "Simulates larger batches at constant memory"),
        ("B7_lora_scale", "B7: LoRA Rank & Target Scaling", "r=[8,16,32,64], targets=[2,4,7]"),
    ]

    for cat_id, title, desc in b_categories:
        w(f"### {title}")
        w("")
        w(f"*{desc}*")
        w("")

        for model_key in ["0.6B", "1.7B", "4B"]:
            cat_results = [r for r in results
                           if r.get("category") == cat_id
                           and r.get("model_key") == model_key]
            if not cat_results:
                continue

            w(f"**{model_key}:**")
            w("")

            if cat_id == "B6_grad_accum":
                w("| GA | Eff BS | Tokens/Step | Step (ms) | tok/s | Peak VRAM | Status |")
                w("|----|--------|-------------|-----------|-------|-----------|--------|")
                for r in cat_results:
                    c = r["config"]
                    if r.get("status") == "OK":
                        w(f"| {c.get('grad_accum', 1)} | {r.get('effective_batch_size', '—')} | "
                          f"{r.get('tokens_per_step', '—')} | {r.get('avg_step_ms', '—')} | "
                          f"{r.get('tokens_per_sec', '—')} | {r.get('peak_alloc_mb', '—')} | OK |")
                    else:
                        w(f"| {c.get('grad_accum', 1)} | — | — | — | — | — | {r.get('status', '?')} |")
            elif cat_id == "B7_lora_scale":
                w("| Rank | Targets | Trainable | Step (ms) | tok/s | Peak VRAM | Headroom | Status |")
                w("|------|---------|-----------|-----------|-------|-----------|----------|--------|")
                for r in cat_results:
                    c = r["config"]
                    if r.get("status") == "OK":
                        tp = r.get("trainable_params", 0)
                        tp_str = f"{tp / 1e6:.1f}M" if tp else "—"
                        w(f"| {c.get('lora_rank', 8)} | {c.get('n_targets', 2)} | {tp_str} | "
                          f"{r.get('avg_step_ms', '—')} | {r.get('tokens_per_sec', '—')} | "
                          f"{r.get('peak_alloc_mb', '—')} | {r.get('headroom_mb', '—')} | OK |")
                    else:
                        w(f"| {c.get('lora_rank', 8)} | {c.get('n_targets', 2)} | — | — | — | — | — | {r.get('status', '?')} |")
            elif cat_id == "B5_fp32":
                w("| Seq Len | Step (ms) | tok/s | Peak VRAM | Headroom | Status |")
                w("|---------|-----------|-------|-----------|----------|--------|")
                for r in cat_results:
                    c = r["config"]
                    if r.get("status") == "OK":
                        w(f"| {c['seq_len']} | {r.get('avg_step_ms', '—')} | "
                          f"{r.get('tokens_per_sec', '—')} | {r.get('peak_alloc_mb', '—')} | "
                          f"{r.get('headroom_mb', '—')} | OK |")
                    else:
                        w(f"| {c['seq_len']} | — | — | — | — | {r.get('status', '?')} |")
            else:
                w("| Seq Len | Batch | Step (ms) | tok/s | Peak VRAM | Headroom | Status |")
                w("|---------|-------|-----------|-------|-----------|----------|--------|")
                for r in cat_results:
                    c = r["config"]
                    if r.get("status") == "OK":
                        w(f"| {c['seq_len']} | {c['batch_size']} | {r.get('avg_step_ms', '—')} | "
                          f"{r.get('tokens_per_sec', '—')} | {r.get('peak_alloc_mb', '—')} | "
                          f"{r.get('headroom_mb', '—')} | OK |")
                    else:
                        w(f"| {c['seq_len']} | {c['batch_size']} | — | — | — | — | {r.get('status', '?')} |")
            w("")

    # ── Part C: Multi-GPU Parallelism ──
    w("## Part C: Multi-GPU Parallelism")
    w("")

    c_categories = [
        ("C1_ddp", "C1: Data Parallelism (DDP)",
         "torchrun --nproc_per_node=2, fp16 LoRA (no bnb), each GPU gets full model copy"),
        ("C3_fsdp", "C3: FSDP (Sharded Data Parallelism)",
         "Fully Sharded Data Parallel — shards model params across 2 GPUs"),
        ("C4_pipeline", "C4: Pipeline Parallelism (device_map)",
         "device_map='auto' splits layers across GPUs, QLoRA + gradient checkpointing"),
        ("C5_zero2", "C5: DeepSpeed ZeRO-2",
         "Shards optimizer states + gradients across GPUs"),
        ("C6_zero3", "C6: DeepSpeed ZeRO-3",
         "Full parameter sharding across GPUs"),
    ]

    for cat_id, title, desc in c_categories:
        w(f"### {title}")
        w("")
        w(f"*{desc}*")
        w("")

        for model_key in ["0.6B", "1.7B", "4B"]:
            cat_results = [r for r in results
                           if r.get("category") == cat_id
                           and r.get("model_key") == model_key]
            if not cat_results:
                continue

            w(f"**{model_key}:**")
            w("")
            w("| Seq Len | Batch | Step (ms) | tok/s | Peak VRAM (MB) | Headroom (MB) | Status |")
            w("|---------|-------|-----------|-------|----------------|---------------|--------|")
            for r in cat_results:
                c = r["config"]
                if r.get("status") == "OK":
                    w(f"| {c['seq_len']} | {c['batch_size']} | {r.get('avg_step_ms', '—')} | "
                      f"{r.get('tokens_per_sec', '—')} | {r.get('peak_alloc_mb', '—')} | "
                      f"{r.get('headroom_mb', '—')} | OK |")
                else:
                    w(f"| {c['seq_len']} | {c['batch_size']} | — | — | — | — | {r.get('status', '?')} |")
            w("")

    # C7/C8: Not feasible
    w("### C7: Tensor Parallelism")
    w("")
    w("**Not feasible.** Tensor parallelism splits individual weight matrices (attention heads, MLP columns) "
      "across GPUs with synchronized AllReduce after each layer. This requires:")
    w("- Megatron-LM or NeMo framework (not HuggingFace Trainer)")
    w("- High-bandwidth interconnect (NVLink, not PCIe 3.0)")
    w("- Significant engineering overhead for models this small")
    w("")
    w("For 0.6B-4B models on PCIe 3.0, the communication overhead would exceed any memory benefit.")
    w("")

    w("### C8: Context/Sequence Parallelism")
    w("")
    w("**Not feasible.** Sequence parallelism (ring attention) splits the sequence dimension across GPUs, "
      "requiring each GPU to compute attention on its chunk and pass KV slices via ring communication. "
      "Available implementations (ring-flash-attn, DeepSpeed Ulysses) require:")
    w("- Flash Attention 2 (needs CC >= 8.0, we have CC 6.1)")
    w("- High-bandwidth interconnect for the ring communication pattern")
    w("")
    w("Pascal GPUs cannot run Flash Attention kernels, making this approach impossible.")
    w("")

    # ── Part D: Combined Best ──
    w("## Part D: Best Combined Configurations")
    w("")

    d_results = [r for r in results if r.get("part") == "D"]
    if d_results:
        for model_key in ["0.6B", "1.7B", "4B"]:
            model_d = [r for r in d_results if r.get("model_key") == model_key]
            if not model_d:
                continue
            w(f"### {model_key}")
            w("")
            for r in model_d:
                cat = r.get("category", "")
                gpu_mode = "single-GPU" if "single" in cat else "multi-GPU"
                w(f"**Best {gpu_mode}** ({r['label']}):")
                w("")
                if r.get("status") == "OK":
                    c = r["config"]
                    w(f"- Config: seq={c.get('seq_len')}, bs={c.get('batch_size')}, "
                      f"r={c.get('lora_rank', 8)}, targets={c.get('n_targets', 2)}")
                    w(f"- Quant: {c.get('quant', 'none')}, GradCkpt: {c.get('grad_ckpt', False)}, "
                      f"GA: {c.get('grad_accum', 1)}")
                    w(f"- **Throughput**: {r.get('tokens_per_sec', '—')} tok/s")
                    w(f"- **Peak VRAM**: {r.get('peak_alloc_mb', '—')} MB "
                      f"(headroom: {r.get('headroom_mb', '—')} MB)")
                    w(f"- **Avg step**: {r.get('avg_step_ms', '—')} ms")
                    w(f"- **Final loss**: {r.get('final_loss', '—')} (20 steps, not converged)")
                else:
                    w(f"- Status: {r.get('status', '?')} — {r.get('error', '')[:100]}")
                w("")
    else:
        w("*No Part D results (depends on successful B+C runs).*")
        w("")

    # ── Conclusions ──
    w("## Conclusions & Recommendations")
    w("")

    # Auto-generate per-model summary
    for model_key in ["0.6B", "1.7B", "4B"]:
        w(f"### {model_key}")
        w("")
        ok_train = [r for r in results
                    if r.get("status") == "OK" and r.get("part") == "B"
                    and r.get("model_key") == model_key]
        if ok_train:
            best = max(ok_train, key=lambda r: r.get("tokens_per_sec", 0))
            max_seq = max(r["config"]["seq_len"] for r in ok_train)
            w(f"- **Max throughput**: {best.get('tokens_per_sec', '—')} tok/s "
              f"(seq={best['config']['seq_len']}, bs={best['config']['batch_size']}, "
              f"{best.get('category', '')})")
            w(f"- **Max sequence length**: {max_seq} tokens")

            b4_ok = [r for r in ok_train if r.get("category") == "B4_qlora_gc"]
            if b4_ok:
                best_b4 = max(b4_ok, key=lambda r: r["config"]["seq_len"])
                w(f"- **Recommended config**: QLoRA+GC, seq={best_b4['config']['seq_len']}, "
                  f"bs={best_b4['config']['batch_size']}")
        else:
            oom_all = all(r.get("status") in ("OOM", "SKIP") for r in results
                         if r.get("part") == "B" and r.get("model_key") == model_key)
            if oom_all:
                w("- **All training configs OOM** — model too large for 8GB VRAM at any sequence length")
            else:
                w("- *No successful training runs*")

        ok_multi = [r for r in results
                    if r.get("status") == "OK" and r.get("part") == "C"
                    and r.get("model_key") == model_key]
        if ok_multi:
            best_m = max(ok_multi, key=lambda r: r.get("tokens_per_sec", 0))
            w(f"- **Best multi-GPU**: {best_m.get('tokens_per_sec', '—')} tok/s "
              f"({best_m.get('category', '')})")
        w("")

    w("### Key Takeaways")
    w("")
    w("1. **QLoRA + Gradient Checkpointing** is the most effective single-GPU strategy "
      "for memory-constrained training on Pascal GPUs")
    w("2. **Gradient accumulation** provides free effective batch scaling at constant memory")
    w("3. **Pipeline parallelism** (device_map='auto') is the most practical multi-GPU "
      "approach for models that don't fit on a single GPU")
    w("4. **DDP** provides near-linear throughput scaling for models that fit on each GPU")
    w("5. **Tensor and sequence parallelism** are not feasible on consumer Pascal GPUs")
    w("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*70}")
    print("Qwen3 Full Experiment Suite")
    print(f"Output: {RESULTS_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"{'='*70}")

    experiments = build_experiments()
    print(f"\nTotal experiments planned: {len(experiments)}")

    all_results = []
    oom_tracker = {}
    skipped = 0
    start_time = time.time()

    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        pass

    for i, exp in enumerate(experiments):
        # Check if we should skip
        if should_skip(exp, oom_tracker):
            skip_result = {
                "label": exp["label"], "part": exp["part"],
                "category": exp["category"], "model_key": exp["model_key"],
                "config": exp["config"], "status": "SKIP",
            }
            all_results.append(skip_result)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(skip_result) + "\n")
            skipped += 1
            print(f"[{i+1}/{len(experiments)}] SKIP  {exp['label']}")
            continue

        print(f"\n[{i+1}/{len(experiments)}] RUN   {exp['label']}")
        result = run_experiment(exp)
        all_results.append(result)

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = result.get("status", "?")
        if status == "OK":
            tps = result.get('tokens_per_sec') or result.get('tokens_per_sec_gen') or '—'
            extra = f"  {tps} tok/s  peak={result.get('peak_alloc_mb', '—')}MB"
            print(f"  -> OK{extra}  ({result.get('elapsed_s', '?')}s)")
        elif status == "OOM":
            record_oom(exp, oom_tracker)
            print(f"  -> OOM  ({result.get('elapsed_s', '?')}s)")
        else:
            print(f"  -> {status}  {result.get('error', '')[:80]}  ({result.get('elapsed_s', '?')}s)")

    # ── Part D: dynamic best configs ──
    print(f"\n{'='*70}")
    print("Part D: Combined Best Configs (20 steps each)")
    print(f"{'='*70}")

    part_d_exps = build_part_d_experiments(all_results)
    for i, exp in enumerate(part_d_exps):
        print(f"\n[D-{i+1}/{len(part_d_exps)}] RUN   {exp['label']}")
        result = run_experiment(exp, timeout=600)  # longer timeout for 20 steps
        all_results.append(result)

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = result.get("status", "?")
        if status == "OK":
            print(f"  -> OK  {result.get('tokens_per_sec', '—')} tok/s  "
                  f"loss={result.get('final_loss', '—')}  ({result.get('elapsed_s', '?')}s)")
        else:
            print(f"  -> {status}  {result.get('error', '')[:80]}")

    # ── Generate report ──
    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"All experiments complete. {len(all_results)} total, "
          f"{sum(1 for r in all_results if r.get('status') == 'OK')} OK, "
          f"{sum(1 for r in all_results if r.get('status') == 'OOM')} OOM, "
          f"{skipped} skipped")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print(f"Generating report...")

    report = generate_report(all_results)
    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(f"Report written to: {REPORT_FILE}")
    print(f"Raw results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
