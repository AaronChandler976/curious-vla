"""Latency and memory of one SmolVLA checkpoint at inference, on the current device.

  python hpc/profile_inference.py --policy runs/<run>/checkpoints/last/pretrained_model [--iters 100] [--out FILE.json]

Times the full pipeline the LIBERO eval runs per decision (preprocessor, then predicting one
50-step action chunk with the flow-matching expert's 10 denoising steps) on a synthetic
observation of the real shapes (two 3x256x256 cameras, 8-D state, an instruction). Reports
mean/p50/p95 latency per chunk, the amortized latency per control step for several
n_action_steps, peak GPU memory, and parameter counts (total and trainable under the
checkpoint's own freezing settings).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(a.policy).to(a.device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=a.policy,
                                      preprocessor_overrides={"device_processor": {"device": a.device}})
    cfg = policy.config
    # Trainable parameters as the fine-tune saw them (SmolVLA freezes the VLM when train_expert_only).
    policy.train()
    if hasattr(policy.model, "vlm_with_expert") and hasattr(policy.model.vlm_with_expert, "set_requires_grad"):
        policy.model.vlm_with_expert.set_requires_grad()
    policy.eval()
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    rng = np.random.default_rng(0)
    def obs():
        return {"observation.images.image": torch.rand(1, 3, 256, 256),
                "observation.images.image2": torch.rand(1, 3, 256, 256),
                "observation.state": torch.as_tensor(rng.normal(size=(1, 8)), dtype=torch.float32),
                "task": ["put the alphabet soup in the basket"]}

    sync = (lambda: torch.cuda.synchronize()) if a.device.startswith("cuda") else (lambda: None)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times, pre_times = [], []
    with torch.no_grad():
        for i in range(a.warmup + a.iters):
            o = obs()
            sync(); t0 = time.perf_counter()
            b = pre(o)
            sync(); t1 = time.perf_counter()
            chunk = policy.predict_action_chunk(b)
            sync(); t2 = time.perf_counter()
            if i >= a.warmup:
                times.append((t2 - t0) * 1000); pre_times.append((t1 - t0) * 1000)
    t = np.array(times)
    res = {"policy": a.policy, "device": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else a.device,
           "chunk_shape": list(chunk.shape), "chunk_size": cfg.chunk_size, "denoise_steps": cfg.num_steps,
           "latency_ms_mean": float(t.mean()), "latency_ms_p50": float(np.percentile(t, 50)),
           "latency_ms_p95": float(np.percentile(t, 95)), "preprocess_ms_mean": float(np.mean(pre_times)),
           "amortized_ms_per_step": {n: float(t.mean() / n) for n in (1, 10, 25, 50)},
           "max_control_hz_open_loop_50": float(1000 / (t.mean() / 50)),
           "peak_gpu_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9) if a.device.startswith("cuda") else None,
           "params_total_m": n_total / 1e6, "params_trainable_m": n_train / 1e6, "iters": a.iters}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
