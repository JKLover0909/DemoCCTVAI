#!/usr/bin/env python3
"""
Phase 5: Benchmark Report Generator

Reads:
  results/baseline_metrics.csv   — OpenCV + PyTorch YOLO pipeline
  results/deepstream_metrics.csv — DeepStream NVDEC + TensorRT pipeline

Generates:
  results/benchmark_report.png   — 4-panel comparison figure
  stdout                         — summary comparison table

Usage:
    python src/generate_report.py
    python src/generate_report.py --baseline results/baseline_metrics.csv
                                  --deepstream results/deepstream_metrics.csv
                                  --out results/benchmark_report.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no display required)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================================
# LOAD CSV
# ============================================================================
def load_csv(path: Path) -> dict:
    """Load metrics CSV. Returns dict of column_name → list[float]."""
    if not path.exists():
        return {}
    cols = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


# ============================================================================
# SUMMARY TABLE (stdout)
# ============================================================================
def print_table(baseline: dict, deepstream: dict):
    def stat(col_data, col_name):
        if not col_data or col_name not in col_data:
            return "N/A", "N/A", "N/A"
        vals = np.array(col_data[col_name])
        return f"{np.mean(vals):.1f}", f"{np.min(vals):.1f}", f"{np.max(vals):.1f}"

    def row(label, col, unit=""):
        bm, bl, bh = stat(baseline, col)
        dm, dl, dh = stat(deepstream, col)

        # Compute delta % vs baseline (positive = DS is better)
        try:
            bv = float(bm)
            dv = float(dm)
            if bv != 0:
                delta = (dv - bv) / abs(bv) * 100.0
                arrow = "▲" if delta > 0 else "▼"
                delta_str = f"{arrow} {abs(delta):.0f}%"
            else:
                delta_str = "—"
        except (ValueError, ZeroDivisionError):
            delta_str = "—"

        print(
            f"  {label:<22} "
            f"{bm:>8}{unit}  ({bl}–{bh})   "
            f"{dm:>8}{unit}  ({dl}–{dh})   "
            f"{delta_str}"
        )

    have_baseline = bool(baseline)
    have_ds = bool(deepstream)

    if not have_baseline and not have_ds:
        print("No CSV data found. Run benchmarks first.")
        return

    print()
    print("=" * 80)
    print("  BENCHMARK COMPARISON REPORT")
    print("  Baseline: OpenCV FFmpeg decode + PyTorch YOLO FP16")
    print("  DeepStream: NVDEC (GPU decode) + TensorRT YOLO FP16")
    print("=" * 80)
    print(
        f"  {'Metric':<22} "
        f"{'Baseline':>9}  (min–max)   "
        f"{'DeepStream':>9}  (min–max)   "
        f"Delta (DS vs Baseline)"
    )
    print("-" * 80)

    # FPS: higher is better. Delta: DS FPS - Baseline FPS as % of Baseline.
    row("FPS",            "fps",          " fps")
    row("Latency",        "latency_ms",   " ms  ")
    row("CPU Usage",      "cpu_pct",      "%    ")
    row("GPU Utilisation","gpu_util_pct", "%    ")
    row("GPU Memory",     "gpu_mem_mb",   " MB  ")

    if baseline and "infer_ms" in baseline and any(float(v) > 0 for v in (baseline.get("infer_ms") or ["0"])):
        row("Infer time/frame",  "infer_ms", " ms  ")

    print("-" * 80)
    print()
    print("  Notes:")
    print("  • FPS:        higher is better")
    print("  • Latency:    lower is better")
    print("  • CPU Usage:  lower is better (less host load = more headroom for scale)")
    print("  • GPU util:   higher is better (better utilisation of GPU)")
    print("  • Delta:      ▲ means DeepStream outperforms baseline")
    print("=" * 80)
    print()


# ============================================================================
# PLOT
# ============================================================================
def make_plot(baseline: dict, deepstream: dict, out_path: Path):
    if not MATPLOTLIB_OK:
        print("WARNING: matplotlib not installed — skipping plot generation.")
        print("         Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "DeepStream 8.0 vs OpenCV+PyTorch Baseline — PPE Benchmark (1 × RTSP H264 Camera)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)

    colors = {"baseline": "#4C72B0", "deepstream": "#DD8452"}

    # ---- Helper: time axis ----
    def time_axis(data):
        if "timestamp_s" in data:
            return np.array(data["timestamp_s"])
        return np.arange(len(next(iter(data.values()))))

    # ------------------------------------------------------------------
    # Panel 1: FPS over time
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if baseline and "fps" in baseline:
        t = time_axis(baseline)
        ax1.plot(t, baseline["fps"], color=colors["baseline"], lw=1.5,
                 label=f"Baseline  (avg {np.mean(baseline['fps']):.1f})")
    if deepstream and "fps" in deepstream:
        t = time_axis(deepstream)
        ax1.plot(t, deepstream["fps"], color=colors["deepstream"], lw=1.5,
                 label=f"DeepStream (avg {np.mean(deepstream['fps']):.1f})")
    ax1.set_title("Throughput (FPS) over time", fontsize=11)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("FPS")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel 2: Latency histogram (end-to-end, ms)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    if baseline and "latency_ms" in baseline:
        ax2.hist(
            baseline["latency_ms"], bins=30, alpha=0.7,
            color=colors["baseline"],
            label=f"Baseline  (μ={np.mean(baseline['latency_ms']):.0f}ms)",
        )
    if deepstream and "latency_ms" in deepstream:
        ax2.hist(
            deepstream["latency_ms"], bins=30, alpha=0.7,
            color=colors["deepstream"],
            label=f"DeepStream (μ={np.mean(deepstream['latency_ms']):.0f}ms)",
        )
    ax2.set_title("Latency distribution (ms)", fontsize=11)
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Frequency (1-second samples)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: CPU vs GPU utilisation — grouped bar chart
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    metrics_labels = ["CPU %", "GPU %"]
    x = np.arange(len(metrics_labels))
    width = 0.35

    def avg_or_nan(data, key):
        if data and key in data:
            return np.mean(data[key])
        return 0.0

    b_vals = [avg_or_nan(baseline, "cpu_pct"), avg_or_nan(baseline, "gpu_util_pct")]
    d_vals = [avg_or_nan(deepstream, "cpu_pct"), avg_or_nan(deepstream, "gpu_util_pct")]

    bars_b = ax3.bar(x - width / 2, b_vals, width, label="Baseline",
                     color=colors["baseline"], alpha=0.85)
    bars_d = ax3.bar(x + width / 2, d_vals, width, label="DeepStream",
                     color=colors["deepstream"], alpha=0.85)

    for bar in list(bars_b) + list(bars_d):
        h = bar.get_height()
        if h > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9,
            )

    ax3.set_title("Avg CPU & GPU Utilisation", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_labels)
    ax3.set_ylabel("Utilisation (%)")
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Panel 4: Summary text card
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    lines = [
        "SUMMARY",
        "",
    ]

    def delta_str(b_data, d_data, key, label, unit="", higher_better=True):
        if not b_data or not d_data:
            return f"  {label}: N/A"
        if key not in b_data or key not in d_data:
            return f"  {label}: N/A"
        bv = np.mean(b_data[key])
        dv = np.mean(d_data[key])
        if bv == 0:
            return f"  {label}: baseline={dv:.1f}{unit}"
        delta_pct = (dv - bv) / abs(bv) * 100.0
        if higher_better:
            symbol = "▲" if delta_pct > 0 else "▼"
        else:
            symbol = "▼" if delta_pct > 0 else "▲"  # lower is better
        return (
            f"  {label:<18} "
            f"B:{bv:6.1f}{unit}  DS:{dv:6.1f}{unit}  "
            f"{symbol}{abs(delta_pct):.0f}%"
        )

    lines.append(delta_str(baseline, deepstream, "fps", "FPS", " fps", higher_better=True))
    lines.append(delta_str(baseline, deepstream, "latency_ms", "Latency", " ms", higher_better=False))
    lines.append(delta_str(baseline, deepstream, "cpu_pct", "CPU", "%", higher_better=False))
    lines.append(delta_str(baseline, deepstream, "gpu_util_pct", "GPU Util", "%", higher_better=True))
    lines.append(delta_str(baseline, deepstream, "gpu_mem_mb", "GPU Mem", " MB", higher_better=False))

    lines += [
        "",
        "▲ = DeepStream better   ▼ = Baseline better",
        "",
        "Key advantages of DeepStream:",
        "  • NVDEC: GPU decode → lower CPU load",
        "  • TensorRT FP16: optimised engine",
        "  • Async pipeline: stable FPS",
        "  • Native scale to 50+ cameras",
        "    with same GPU architecture",
    ]

    text = "\n".join(lines)
    ax4.text(
        0.05, 0.95, text,
        transform=ax4.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", alpha=0.8),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📊 Report saved to: {out_path}")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Generate benchmark comparison report")
    p.add_argument(
        "--baseline",
        default=str(RESULTS_DIR / "baseline_metrics.csv"),
        help="Baseline CSV path",
    )
    p.add_argument(
        "--deepstream",
        default=str(RESULTS_DIR / "deepstream_metrics.csv"),
        help="DeepStream CSV path",
    )
    p.add_argument(
        "--out",
        default=str(RESULTS_DIR / "benchmark_report.png"),
        help="Output PNG path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    b_path = Path(args.baseline)
    d_path = Path(args.deepstream)
    o_path = Path(args.out)

    if not b_path.exists():
        print(f"WARNING: Baseline CSV not found: {b_path}")
    if not d_path.exists():
        print(f"WARNING: DeepStream CSV not found: {d_path}")

    baseline = load_csv(b_path)
    deepstream = load_csv(d_path)

    print_table(baseline, deepstream)
    make_plot(baseline, deepstream, o_path)


if __name__ == "__main__":
    main()
