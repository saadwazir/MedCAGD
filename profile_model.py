# profile_model.py
# =======================================================
# Inference-only profiling script
# Reports:
#   - Batch size = 1 (latency)
#   - Batch size = custom / cfgs.py batch (throughput)
# =======================================================

import os
import torch
from contextlib import nullcontext

from thop import profile
from fvcore.nn import FlopCountAnalysis

from cfgs import *
from model import build_model

# =======================================================
# USER-CONTROLLED VARIABLES (EDIT ONLY THESE)
# =======================================================

# ---- Precision mode ----
USE_AMP = 0                 # 0 = FP32, 1 = AMP (autocast)

# ---- Batch size override ----
CUSTOM_BATCH_SIZE = ""      # "" → use batch_size from cfgs.py, else int (e.g., 4, 8)

# ---- GPU override ----
GPU_OVERRIDE = ""           # "" → use gpu_list from cfgs.py, else e.g. "0" or "0,1"

# ---- Profiling settings ----
WARMUP_ITERS = 20
BENCH_ITERS = 50
OUTPUT_FILE = "flops.txt"


# =======================================================
# INTERNAL: profile once for a given batch size
# =======================================================
def profile_once(model, device, batch_size, log, amp_ctx):

    x = torch.randn(batch_size, 3, H, W, device=device)

    log(f"Batch size       : {batch_size}")

    # ---------------- Parameters ----------------
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total Params     : {total_params:,}")

    # ---------------- MACs (THOP) ----------------
    with torch.no_grad(), amp_ctx:
        macs, _ = profile(model, inputs=(x,), verbose=False)
    log(f"MACs             : {macs / 1e9:.3f} G")

    # ---------------- FLOPs (FVCore) ----------------
    with torch.no_grad(), amp_ctx:
        flops = FlopCountAnalysis(model, x).total()
    log(f"FLOPs            : {flops / 1e9:.3f} GFLOPs")

    # ---------------- GPU Warm-up ----------------
    with torch.no_grad(), amp_ctx:
        for _ in range(WARMUP_ITERS):
            _ = model(x)

    torch.cuda.synchronize()

    # ---------------- Latency / Throughput ----------------
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    with torch.no_grad(), amp_ctx:
        for _ in range(BENCH_ITERS):
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

    avg_batch_latency = sum(times) / len(times)
    latency_per_image = avg_batch_latency / batch_size
    throughput = 1000.0 / latency_per_image

    log(f"Avg batch latency : {avg_batch_latency:.3f} ms")
    log(f"Latency / image   : {latency_per_image:.3f} ms")
    log(f"Throughput        : {throughput:.2f} images/sec")


# =======================================================
# MAIN
# =======================================================
def main():

    # ---------------------------------------------------
    # Resolve GPU list
    # ---------------------------------------------------
    effective_gpu_list = GPU_OVERRIDE if GPU_OVERRIDE.strip() else gpu_list
    os.environ["CUDA_VISIBLE_DEVICES"] = effective_gpu_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------
    # Resolve batch size
    # ---------------------------------------------------
    if CUSTOM_BATCH_SIZE == "" or CUSTOM_BATCH_SIZE is None:
        effective_custom_bs = batch_size   # from cfgs.py
    else:
        effective_custom_bs = int(CUSTOM_BATCH_SIZE)

    # ---------------------------------------------------
    # AMP context
    # ---------------------------------------------------
    amp_ctx = (
        torch.cuda.amp.autocast(enabled=True)
        if USE_AMP == 1 and device.type == "cuda"
        else nullcontext()
    )

    # ---------------------------------------------------
    # Logging helper
    # ---------------------------------------------------
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    # ---------------------------------------------------
    # Header
    # ---------------------------------------------------
    log("=" * 90)
    log("INFERENCE-ONLY MODEL PROFILING")
    log(f"Visible GPUs     : {effective_gpu_list}")
    log(f"Device           : {device}")
    log(f"Input resolution : {H} x {W}")
    log(f"AMP autocast     : {'ON' if USE_AMP == 1 else 'OFF'}")
    log("=" * 90)

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = build_model().to(device)
    model.eval()

    # ===================================================
    # CASE A: Batch size = 1 (Latency)
    # ===================================================
    log("\n[CASE A] Batch size = 1 (Latency-critical)")
    log("-" * 90)
    profile_once(model, device, 1, log, amp_ctx)

    # ===================================================
    # CASE B: Custom / cfg batch size (Throughput)
    # ===================================================
    log(f"\n[CASE B] Batch size = {effective_custom_bs} (Throughput-oriented)")
    log("-" * 90)
    profile_once(model, device, effective_custom_bs, log, amp_ctx)

    log("=" * 90)

    # ---------------------------------------------------
    # Save results
    # ---------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\n[INFO] Results saved to → {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
