# =============================================================================
# Makefile — PPE DeepStream vs Baseline Benchmark POC
#
# WORKFLOW (one-time setup):
#   1.  make inspect        ← find out PPE class names from best.pt
#   2.  make build-ds       ← build DeepStream Docker image (takes ~10 min)
#   3.  make export         ← export best.pt → ONNX → TensorRT engine
#
# BENCHMARKS (run once both are set up):
#   4a. make baseline       ← OpenCV + PyTorch pipeline (host, no Docker)
#   4b. make deepstream     ← DeepStream pipeline (Docker)
#
# REPORT:
#   5.  make report         ← generate benchmark_report.png + summary table
#
# QUICK ALL-IN-ONE (after setup):
#   make bench-all
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Configurable variables (override on CLI: make baseline DURATION=60)
# ---------------------------------------------------------------------------
PYTHON     ?= python3
PROJECT    := $(shell pwd)
# best.pt: PCB detector (nc=1). Both models in this repo are fine-tuned on PCB data.
# For this benchmark the model subject does not matter — we compare pipeline metrics (FPS/latency/CPU/GPU).
MODEL      ?= models/best.pt
RTSP_URL   ?= rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101
DURATION   ?= 120
WARMUP     ?= 30
DS_IMAGE   ?= deepstream-ppe:latest

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "  PPE DeepStream vs Baseline Benchmark — Make targets"
	@echo ""
	@echo "  SETUP:"
	@echo "    make inspect      Inspect best.pt → print class names, write labels.txt"
	@echo "    make build-ds     Build DeepStream Docker image (first time ~10 min)"
	@echo "    make export       Export best.pt → ONNX → TensorRT .engine (needs build-ds)"
	@echo ""
	@echo "  BENCHMARKS:"
	@echo "    make baseline     Run OpenCV+PyTorch baseline (writes results/baseline_metrics.csv)"
	@echo "    make deepstream   Run DeepStream pipeline in Docker (writes results/deepstream_metrics.csv)"
	@echo "    make bench-all    Run both benchmarks sequentially then generate report"
	@echo ""
	@echo "  REPORT:"
	@echo "    make report       Generate benchmark_report.png + summary table"
	@echo ""
	@echo "  UTILS:"
	@echo "    make clean        Remove results/*.csv and results/*.png"
	@echo "    make results-dir  Create results/ directory"
	@echo ""
	@echo "  Override variables:"
	@echo "    RTSP_URL='rtsp://...'   Camera URL"
	@echo "    DURATION=120            Measurement duration (seconds)"
	@echo "    WARMUP=30               Warmup duration (seconds)"
	@echo "    MODEL=models/best.pt    YOLO model path"
	@echo "    PYTHON=python3          Python executable"
	@echo "    DS_IMAGE=deepstream-ppe:latest  Docker image name"
	@echo ""

# ---------------------------------------------------------------------------
# results directory
# ---------------------------------------------------------------------------
.PHONY: results-dir
results-dir:
	@mkdir -p results

# ---------------------------------------------------------------------------
# SETUP: inspect model
# ---------------------------------------------------------------------------
.PHONY: inspect
inspect:
	@echo ""
	@echo "=== Phase 0: Inspecting model: $(MODEL) ==="
	@echo ""
	$(PYTHON) src/inspect_model.py --model $(MODEL)
	@echo ""
	@echo "Update deepstream/config/config_infer_primary.txt:"
	@echo "  Set num-detected-classes= to the number of classes shown above."
	@echo ""

# ---------------------------------------------------------------------------
# SETUP: build DeepStream Docker image
# ---------------------------------------------------------------------------
.PHONY: build-ds
build-ds:
	@echo ""
	@echo "=== Building DeepStream Docker image: $(DS_IMAGE) ==="
	@echo "    Base: nvcr.io/nvidia/deepstream:8.0-samples-unified"
	@echo "    This includes DeepStream-Yolo YOLO11 parser compilation."
	@echo "    First build takes ~10-15 minutes."
	@echo ""
	docker build --network=host -t $(DS_IMAGE) deepstream/
	@echo ""
	@echo "✅ Docker image ready: $(DS_IMAGE)"
	@echo ""

# ---------------------------------------------------------------------------
# SETUP: export model → TensorRT engine
# ---------------------------------------------------------------------------
.PHONY: export
export:
	@echo ""
	@echo "=== Phase 2: Exporting $(MODEL) → ONNX → TensorRT engine ==="
	@echo ""
	bash scripts/export_to_trt.sh --model $(basename $(notdir $(MODEL)) .pt)
	@echo ""

# ---------------------------------------------------------------------------
# BENCHMARK: baseline (OpenCV + PyTorch, runs on host)
# ---------------------------------------------------------------------------
.PHONY: baseline
baseline: results-dir
	@echo ""
	@echo "=== Phase 1: Baseline Benchmark ==="
	@echo "    Pipeline: OpenCV FFmpeg (CPU decode) + PyTorch YOLO FP16 (GPU infer)"
	@echo "    RTSP URL: $(RTSP_URL)"
	@echo "    Warmup:   $(WARMUP)s   Measure: $(DURATION)s"
	@echo "    Output:   results/baseline_metrics.csv"
	@echo ""
	$(PYTHON) src/baseline_benchmark.py \
		--url "$(RTSP_URL)" \
		--model $(MODEL) \
		--warmup $(WARMUP) \
		--duration $(DURATION)
	@echo ""
	@echo "✅ Baseline CSV: results/baseline_metrics.csv"

# ---------------------------------------------------------------------------
# BENCHMARK: DeepStream (runs in Docker container)
# ---------------------------------------------------------------------------
.PHONY: deepstream
deepstream: results-dir
	@echo ""
	@echo "=== Phase 3: DeepStream Benchmark ==="
	@echo "    Pipeline: NVDEC (GPU decode) + TensorRT YOLO FP16"
	@echo "    Docker:   $(DS_IMAGE)"
	@echo "    RTSP URL: $(RTSP_URL)"
	@echo "    Warmup:   $(WARMUP)s   Measure: $(DURATION)s"
	@echo "    Output:   results/deepstream_metrics.csv"
	@echo ""
	RTSP_URL="$(RTSP_URL)" DURATION="$(DURATION)" WARMUP="$(WARMUP)" \
		bash deepstream/run_benchmark.sh
	@echo ""
	@echo "✅ DeepStream CSV: results/deepstream_metrics.csv"

# ---------------------------------------------------------------------------
# Run both benchmarks sequentially → report
# ---------------------------------------------------------------------------
.PHONY: bench-all
bench-all: baseline deepstream report
	@echo ""
	@echo "✅ Full benchmark complete."
	@echo "   Open results/benchmark_report.png to view comparison."

# ---------------------------------------------------------------------------
# REPORT: generate PNG + summary table
# ---------------------------------------------------------------------------
.PHONY: report
report: results-dir
	@echo ""
	@echo "=== Phase 5: Generating Report ==="
	@echo ""
	$(PYTHON) src/generate_report.py
	@echo ""

# ---------------------------------------------------------------------------
# Utility: clean results
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Removing results/*.csv and results/*.png ..."
	rm -f results/*.csv results/*.png
	@echo "Done."

# ---------------------------------------------------------------------------
# Utility: show current results
# ---------------------------------------------------------------------------
.PHONY: show-results
show-results:
	@echo ""
	@echo "=== CSV files in results/ ==="
	@ls -lh results/*.csv 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "=== Latest summary ==="
	$(PYTHON) src/generate_report.py 2>/dev/null || echo "  (run make report to generate)"
	@echo ""
