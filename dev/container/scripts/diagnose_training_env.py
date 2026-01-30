#!/usr/bin/env python3
"""Diagnose PyTorch/HF training environment for known issues.

Checks GPU hardware, library versions, and flags known incompatibilities.
Run inside the training environment (container, venv, etc.).

Usage: python diagnose_training_env.py [--json]
"""
import json as json_mod
import sys


def check_torch():
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "cc_tuple": (props.major, props.minor),
                })
        return info
    except ImportError:
        return {"error": "torch not installed"}


def check_libraries():
    libs = {}
    for name in ["transformers", "peft", "bitsandbytes", "accelerate", "datasets",
                  "deepspeed", "flash_attn", "trl"]:
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            libs[name] = None
    return libs


def check_env_vars():
    import os
    relevant = {}
    for var in ["CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
                "TORCH_USE_CUDA_DSA", "NCCL_DEBUG", "NCCL_P2P_DISABLE",
                "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                "TRANSFORMERS_CACHE", "HF_HOME", "HF_HUB_OFFLINE"]:
        val = os.environ.get(var)
        if val is not None:
            relevant[var] = val
    return relevant


def diagnose(torch_info, libs, env_vars):
    """Return list of (severity, message) tuples."""
    issues = []

    if not torch_info.get("cuda_available"):
        issues.append(("CRITICAL", "CUDA not available — training will use CPU"))
        return issues

    gpus = torch_info.get("gpus", [])
    n_gpu = len(gpus)

    # Check compute capability
    for gpu in gpus:
        cc = gpu.get("cc_tuple", (0, 0))
        if cc < (7, 0):
            issues.append(("WARN", f"GPU {gpu['index']} ({gpu['name']}) CC {gpu['compute_capability']}: "
                          "no bf16, no Flash Attention 2. Use fp16=True only."))
        if cc < (8, 0):
            issues.append(("INFO", f"GPU {gpu['index']} CC {gpu['compute_capability']}: "
                          "Flash Attention 2 requires CC >= 8.0 (Ampere+)"))

    # bnb + multi-GPU DataParallel issue
    if n_gpu > 1 and libs.get("bitsandbytes"):
        cvd = env_vars.get("CUDA_VISIBLE_DEVICES")
        if cvd is None or len(cvd.split(",")) > 1:
            issues.append(("CRITICAL", f"Multi-GPU ({n_gpu}) with bitsandbytes: "
                          "HF Trainer will wrap 4-bit models in DataParallel, causing "
                          "'illegal memory access'. Set CUDA_VISIBLE_DEVICES=0 for "
                          "single-GPU training, or use subprocess isolation."))

    # CUDA_LAUNCH_BLOCKING + bnb
    if env_vars.get("CUDA_LAUNCH_BLOCKING") == "1" and libs.get("bitsandbytes"):
        issues.append(("WARN", "CUDA_LAUNCH_BLOCKING=1 with bitsandbytes: "
                      "bnb 4-bit kernels become extremely slow (10-100x). "
                      "Training may appear to hang. Remove for production runs."))

    # peft version check
    if libs.get("peft"):
        issues.append(("INFO", "peft installed: remember get_peft_model() drops "
                      "hf_device_map from base model. Restore it after wrapping "
                      "if using device_map with multi-GPU."))

    # Memory warnings
    for gpu in gpus:
        if gpu["total_memory_gb"] < 10:
            issues.append(("INFO", f"GPU {gpu['index']}: {gpu['total_memory_gb']}GB — "
                          "use QLoRA (4-bit) + gradient checkpointing + paged_adamw_8bit "
                          "for models > 1B params."))

    return issues


def main():
    output_json = "--json" in sys.argv

    torch_info = check_torch()
    libs = check_libraries()
    env_vars = check_env_vars()
    issues = diagnose(torch_info, libs, env_vars)

    if output_json:
        print(json_mod.dumps({
            "torch": torch_info,
            "libraries": libs,
            "env_vars": env_vars,
            "issues": [{"severity": s, "message": m} for s, m in issues],
        }, indent=2))
        return

    print("=" * 60)
    print("TRAINING ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)

    print("\n## PyTorch")
    for k, v in torch_info.items():
        if k != "gpus":
            print(f"  {k}: {v}")
    for gpu in torch_info.get("gpus", []):
        print(f"  GPU {gpu['index']}: {gpu['name']} | {gpu['total_memory_gb']}GB | CC {gpu['compute_capability']}")

    print("\n## Libraries")
    for name, ver in libs.items():
        status = ver if ver else "not installed"
        print(f"  {name}: {status}")

    print("\n## Environment Variables")
    if env_vars:
        for k, v in env_vars.items():
            print(f"  {k}={v}")
    else:
        print("  (none set)")

    print(f"\n## Issues ({len(issues)} found)")
    for severity, msg in issues:
        icon = {"CRITICAL": "!!!", "WARN": " ! ", "INFO": " i "}.get(severity, "   ")
        print(f"  [{icon}] {msg}")

    if not issues:
        print("  No known issues detected.")

    # Exit code: 2 if critical, 1 if warnings, 0 if clean
    severities = {s for s, _ in issues}
    if "CRITICAL" in severities:
        sys.exit(2)
    elif "WARN" in severities:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
