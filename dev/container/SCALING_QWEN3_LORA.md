# Scaling Qwen3-0.6B LoRA Training on Consumer GPUs

Empirical results from pushing LoRA fine-tuning of Qwen3-0.6B to its limits on 2x NVIDIA GTX 1070 Ti (8GB VRAM each), using a Ray-based container environment.

## Hardware

- **GPUs**: 2x NVIDIA GeForce GTX 1070 Ti (8,192 MB VRAM, Pascal architecture, compute capability 6.1)
- **Shared memory**: 16 GB (Docker `shm_size`)
- **No bf16 support** — all experiments use fp16

## Setup

All experiments run inside a Docker container built on `nvidia/cuda:12.1.1-devel-ubuntu22.04` with:

- Python 3.10, PyTorch 2.5.1+cu121
- Transformers 5.0.0, PEFT, bitsandbytes 0.49.1
- Ray 2.40.0 (for distributed training integration)
- HuggingFace cache mounted from host (`/data/huggingface`) for model/dataset persistence across runs

The model is loaded from `Qwen/Qwen3-0.6B` (~1.2 GB in fp16, ~0.4 GB in 4-bit NF4).

## Baseline: No Optimizations (fp16)

Starting point — vanilla LoRA (r=8, q\_proj + v\_proj targets) with AdamW optimizer, no memory tricks.

| Seq Len | Max Batch Size | Peak VRAM | Headroom | Throughput |
|---------|---------------|-----------|----------|------------|
| 128     | 8             | 5,885 MB  | 1,565 MB | 1,371 tok/s |
| 256     | 4             | 6,109 MB  | 1,321 MB | 1,309 tok/s |
| 512     | 2             | 6,557 MB  | 837 MB   | 1,211 tok/s |
| **1024**| OOM at bs=1   | —         | —        | —          |

The fp16 baseline tops out at **512 tokens**. At seq=1024, even a single sample overflows 8 GB.

### Where the memory goes

At seq=512, bs=1 (4,064 MB peak):

- **Model weights** (fp16): ~1,200 MB
- **Optimizer states** (AdamW, 2 moments per param): ~2,400 MB for full model, but only LoRA params are optimized (~3 MB optimizer overhead)
- **Activations** (forward pass intermediates for backprop): ~1,500 MB — this is the bottleneck that scales with sequence length
- **Gradients + workspace**: ~400 MB

Activations grow quadratically with sequence length in attention layers (the attention score matrix is `[batch, heads, seq, seq]`). This is why seq=1024 requires roughly 4x the activation memory of seq=512.

## Technique 1: Gradient Checkpointing

**What it does**: Instead of storing all intermediate activations during the forward pass, discard them and recompute on-the-fly during backpropagation. Trades compute time for memory.

**Implementation**: One line — `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`

| Seq Len | Max Batch Size | Peak VRAM | Headroom | Throughput | vs Baseline |
|---------|---------------|-----------|----------|------------|-------------|
| 512     | 2             | 3,603 MB  | 3,727 MB | 907 tok/s  | -25% speed, **-45% memory** |
| **1024**| **1**         | 3,603 MB  | 3,603 MB | 776 tok/s  | **unlocked** |
| 2048    | OOM at bs=1   | —         | —        | —          |             |

Gradient checkpointing cuts peak activation memory by nearly half. The key result: **seq=1024 now fits in 3.6 GB** (less than half the card), at the cost of ~25-35% slower step time due to recomputation.

However, seq=2048 still OOMs — the model weights and optimizer states themselves consume too much of the remaining budget.

## Technique 2: 8-bit Optimizer (AdamW8bit)

**What it does**: Uses bitsandbytes to store AdamW's first and second moment estimates in 8-bit instead of 32-bit, halving optimizer state memory.

**Result**: No improvement over standard AdamW for this model. LoRA only trains ~0.5% of parameters (3.5M out of 620M), so optimizer states are already small (~27 MB for r=8). The savings are negligible compared to activation memory pressure.

**Verdict**: Skip for small LoRA ranks. Becomes relevant at r=64+ with many target modules, where optimizer states reach hundreds of MB.

## Technique 3: QLoRA (4-bit Quantization)

**What it does**: Loads the base model in 4-bit NF4 quantization (Normal Float 4-bit), reducing model weight memory by ~4x. LoRA adapters remain in fp16. Uses double quantization to further compress the quantization constants.

**Implementation**:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

Requires `prepare_model_for_kbit_training()` from PEFT before applying LoRA.

### QLoRA alone (no gradient checkpointing)

| Seq Len | Max Batch Size | Peak VRAM | Headroom | Throughput |
|---------|---------------|-----------|----------|------------|
| 512     | 1             | 4,478 MB  | 3,283 MB | 887 tok/s  |
| 1024    | OOM at bs=1   | —         | —        | —          |

Surprisingly, QLoRA alone doesn't unlock seq=1024. The model is smaller (~400 MB vs ~1,200 MB), but activations still dominate at longer sequences. The 4-bit dequantization during forward pass also adds overhead.

### QLoRA + Gradient Checkpointing (the winning combination)

| Seq Len | Max Batch Size | Peak VRAM | Headroom | Throughput | vs Baseline |
|---------|---------------|-----------|----------|------------|-------------|
| 512     | 4             | 6,436 MB  | 691 MB   | 910 tok/s  | -25% speed, **2x batch** |
| **1024**| **2**         | 6,434 MB  | 689 MB   | 787 tok/s  | **unlocked, bs=2** |
| **2048**| **1**         | 6,431 MB  | 369 MB   | 606 tok/s  | **unlocked** |
| 4096    | OOM at bs=1   | —         | —        | —          |             |

This is the breakthrough combination. QLoRA shrinks the model footprint, and gradient checkpointing eliminates stored activations. Together they push the maximum sequence length to **2048 tokens** — a 4x improvement over the baseline.

At seq=2048, only 369 MB of headroom remains. The 8 GB card is nearly fully utilized.

## Technique 4: Full Stack — QLoRA + GradCkpt + Adam8bit + Larger LoRA

With the memory headroom from QLoRA + gradient checkpointing, we can afford richer LoRA configurations — more target modules and higher rank — for better fine-tuning quality.

### LoRA rank scaling at seq=1024 (4 targets: q, k, v, o projections)

| Rank | Max BS | Trainable Params | Throughput | Peak VRAM |
|------|--------|-----------------|------------|-----------|
| r=16 | 2      | ~3.0M           | 746 tok/s  | 6,444 MB  |
| r=32 | 2      | ~6.0M           | 728 tok/s  | 6,474 MB  |
| r=64 | 1      | ~12.0M          | 671 tok/s  | 4,033 MB  |

### LoRA rank scaling at seq=2048

| Rank | Targets | Max BS | Throughput | Peak VRAM | Headroom |
|------|---------|--------|------------|-----------|----------|
| r=8  | 2 (q,v) | 1     | 606 tok/s  | 6,431 MB  | 369 MB   |
| r=16 | 4 (qkvo)| 1     | 580 tok/s  | 6,448 MB  | 409 MB   |
| r=32 | 4 (qkvo)| 1     | 579 tok/s  | 6,475 MB  | 303 MB   |
| r=64 | 4 (qkvo)| OOM   | —          | —         | —        |

At seq=2048, you can fit up to **r=32 with 4 target modules** — a meaningful LoRA configuration with ~6M trainable parameters. Going to r=64 pushes over the edge.

### 7 target modules (q, k, v, o, gate, up, down projections)

| Seq  | Rank | Max BS | Status |
|------|------|--------|--------|
| 1024 | r=16 | 1      | OK (643 tok/s, 3,991 MB) |
| 1024 | r=16 | 2      | OOM    |
| 1024 | r=32 | 1      | OOM    |

Targeting all 7 linear layers is tight. At seq=1024, only r=16 bs=1 fits. For practical training with 7 targets, stay at seq=512.

## Technique 5: Gradient Accumulation

Gradient accumulation doesn't reduce peak memory — it simulates larger batches by accumulating gradients across multiple micro-batches before an optimizer step. Since each micro-batch runs independently, memory usage equals that of the micro-batch size.

### Effective batch scaling at seq=2048 (QLoRA + GradCkpt + Adam8bit, r=16, 4 targets)

| Grad Accum | Effective BS | Tokens/Step | Step Time | tok/s | Peak VRAM |
|------------|-------------|-------------|-----------|-------|-----------|
| 1          | 1           | 2,048       | 3.5s      | 589   | 6,468 MB  |
| 2          | 2           | 4,096       | 7.0s      | 588   | 6,468 MB  |
| 4          | 4           | 8,192       | 13.9s     | 588   | 6,472 MB  |
| 8          | 8           | 16,384      | 27.8s     | 589   | 6,475 MB  |
| 16         | 16          | 32,768      | 55.7s     | 589   | 6,467 MB  |

Throughput stays perfectly constant at ~589 tok/s regardless of accumulation steps — confirming zero memory overhead. The effective batch size scales linearly with accumulation count.

With ga=16, each optimizer step processes **32,768 tokens** — equivalent to a bs=16 training step that would normally require ~50+ GB VRAM. This is the primary mechanism for training with larger effective batches on memory-constrained hardware.

## Summary: What Works and What Doesn't

### Optimization impact ranking

| Technique | Memory Saved | Speed Impact | Unlocks |
|-----------|-------------|-------------|---------|
| **Gradient Checkpointing** | ~40% of activations | -25% to -35% | seq 512 → 1024 |
| **QLoRA (4-bit NF4)** | ~75% of model weights | -10% to -20% | seq 1024 → 2048 (with GC) |
| **Gradient Accumulation** | 0% | Linear time scaling | Larger effective batch |
| **8-bit Optimizer** | Negligible for LoRA | None | Nothing at low rank |

### Recommended configurations for GTX 1070 Ti (8 GB)

**Best throughput (short context)**:
- seq=128, bs=8, r=8, fp16 baseline — 1,371 tok/s

**Balanced quality training**:
- seq=512, bs=4, r=8, QLoRA + GradCkpt — 910 tok/s
- seq=1024, bs=2, r=32 (4 targets), QLoRA + GradCkpt + Adam8bit — 728 tok/s

**Maximum context length**:
- seq=2048, bs=1, r=32 (4 targets), QLoRA + GradCkpt + Adam8bit, ga=8 — 579 tok/s, effective batch 8

**Unreachable on 8 GB**:
- seq=4096+ at any batch size with any optimization — KV cache alone exceeds available memory

### The hard wall

On 8 GB Pascal GPUs, the absolute memory ceiling for Qwen3-0.6B training is:

- **2048 tokens** maximum sequence length
- **~6.5 GB** peak VRAM utilization (300-700 MB headroom)
- **~600 tok/s** throughput at the limit

To go beyond seq=2048 with this model, you need either more VRAM (16+ GB cards like RTX 3080/4080/A5000) or model parallelism across both GPUs (which adds communication overhead that likely isn't worth it for a 0.6B model).

## Reproducing These Results

```bash
cd dev/container

# Build and start the container
docker compose build
docker compose up -d ray-dev

# Install Ray and bitsandbytes
docker compose exec ray-dev bash dev/container/scripts/build-ray.sh
docker compose exec ray-dev pip install bitsandbytes

# Run the baseline GPU stress test
docker compose exec -w /tmp ray-dev python -u -B \
    /workspace/ray/dev/container/scripts/gpu_stress_test.py

# Run the long sequence optimization test
docker compose exec -w /tmp ray-dev python -u -B \
    /workspace/ray/dev/container/scripts/gpu_long_seq_test.py
```

Raw results are saved to `/data/outputs/qwen3-scaling/gpu_stress_results.jsonl` and `/data/outputs/qwen3-scaling/long_seq_results.jsonl` inside the container (host path: `/mnt/data_infra/shared/outputs/qwen3-scaling/`).
