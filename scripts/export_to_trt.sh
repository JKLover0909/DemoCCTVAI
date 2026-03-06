#!/usr/bin/env bash
# =============================================================================
# scripts/export_to_trt.sh
#
# Step 1: Export best.pt → best.onnx  (runs in host Python venv)
# Step 2: Build best_fp16.engine      (runs inside DeepStream Docker container
#                                      using the container's TensorRT trtexec)
#
# Why build engine INSIDE the container?
#   TensorRT engines are NOT portable. The .engine built on host TRT may differ
#   from the TRT version inside the DeepStream container. Building inside the
#   container guarantees compatibility.
#
# Usage:
#   bash scripts/export_to_trt.sh
#   bash scripts/export_to_trt.sh --fp32          (skip FP16)
#   bash scripts/export_to_trt.sh --batch 4       (build for batch=4)
#   bash scripts/export_to_trt.sh --imgsz 640
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models"
DS_IMAGE="deepstream-ppe:latest"  # must match deepstream/Dockerfile

# Defaults
FP16="true"
BATCH=1
IMGSZ=640
MODEL_NAME="best"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp32)   FP16="false";;
    --batch)  BATCH="$2"; shift;;
    --imgsz)  IMGSZ="$2"; shift;;
    --model)  MODEL_NAME="$2"; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
  shift
done

PT_FILE="$MODEL_DIR/${MODEL_NAME}.pt"
ONNX_FILE="$MODEL_DIR/${MODEL_NAME}.onnx"
if [[ "$FP16" == "true" ]]; then
  ENGINE_FILE="$MODEL_DIR/${MODEL_NAME}_fp16_b${BATCH}.engine"
  PRECISION_FLAG="--fp16"
  PREC_LABEL="FP16"
else
  ENGINE_FILE="$MODEL_DIR/${MODEL_NAME}_fp32_b${BATCH}.engine"
  PRECISION_FLAG=""
  PREC_LABEL="FP32"
fi

echo ""
echo "============================================================"
echo "  EXPORT: ${MODEL_NAME}.pt → ONNX → TensorRT (${PREC_LABEL})"
echo "============================================================"
echo "  Source   : $PT_FILE"
echo "  ONNX out : $ONNX_FILE"
echo "  Engine   : $ENGINE_FILE"
echo "  Batch    : $BATCH"
echo "  Imgsz    : ${IMGSZ}x${IMGSZ}"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 1 — Export .pt → .onnx  (host Python env)
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Exporting to ONNX with ultralytics..."

if [[ ! -f "$PT_FILE" ]]; then
  echo "ERROR: Model not found at $PT_FILE"
  exit 1
fi

# Run ultralytics export. Produces best.onnx in the same directory as the .pt.
python3 - <<PYEOF
from ultralytics import YOLO
import shutil, sys
from pathlib import Path

model_path = "$PT_FILE"
onnx_target = "$ONNX_FILE"
imgsz = $IMGSZ
batch = $BATCH

print(f"  Loading {model_path} ...")
m = YOLO(model_path)
print(f"  Exporting to ONNX (imgsz={imgsz}, batch={batch}) ...")
out = m.export(format="onnx", imgsz=imgsz, batch=batch, simplify=True,
               dynamic=False, opset=17)
out = Path(out)
target = Path(onnx_target)
if out != target:
    shutil.copy(out, target)
    print(f"  Copied {out.name} → {target}")
print(f"  ONNX exported: {target}")
PYEOF

if [[ ! -f "$ONNX_FILE" ]]; then
  echo "ERROR: ONNX export failed — $ONNX_FILE not found."
  exit 1
fi
echo "  ✅ ONNX ready: $ONNX_FILE"

# ---------------------------------------------------------------------------
# STEP 2 — Build TensorRT engine INSIDE DeepStream container
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Building TensorRT engine inside DeepStream container..."
echo "      (This uses the container's trtexec — ensures TRT version compatibility)"
echo ""

# Check Docker image exists
if ! docker image inspect "$DS_IMAGE" &>/dev/null; then
  echo "ERROR: Docker image '$DS_IMAGE' not found."
  echo "       Run 'make build-ds' first to build the image."
  exit 1
fi

TRTEXEC_CMD="/usr/src/tensorrt/bin/trtexec"
WORKSPACE=2048  # MB

INPUT_CHANNEL=3

docker run --rm --gpus all \
  -v "${MODEL_DIR}:/models" \
  "$DS_IMAGE" \
  bash -c "
    set -e
    echo '  TensorRT version:'
    ${TRTEXEC_CMD} --version 2>/dev/null | head -3 || true
    echo ''
    echo '  Building engine...'
    ${TRTEXEC_CMD} \
      --onnx=/models/${MODEL_NAME}.onnx \
      --saveEngine=/models/${MODEL_NAME}_${FP16/true/fp16}_b${BATCH}.engine \
      ${PRECISION_FLAG} \
      --inputIOFormats=fp16:chw \
      --outputIOFormats=fp16:chw \
      --workspace=${WORKSPACE} \
      --iterations=100 \
      --warmUp=1000 \
      --avgRuns=10 \
      --verbose \
      2>&1 | grep -E '(Building|Engine|latency|throughput|Error|error|Timing|ILayer|mean|median)' || \
    ${TRTEXEC_CMD} \
      --onnx=/models/${MODEL_NAME}.onnx \
      --saveEngine=/models/${MODEL_NAME}_${FP16/true/fp16}_b${BATCH}.engine \
      ${PRECISION_FLAG} \
      --workspace=${WORKSPACE}
  "

if [[ ! -f "$ENGINE_FILE" ]]; then
  echo ""
  echo "ERROR: Engine file not found after build: $ENGINE_FILE"
  echo "       Check trtexec output above for errors."
  exit 1
fi

echo ""
echo "============================================================"
echo "  ✅ TensorRT engine ready: $ENGINE_FILE"
echo "     Size: $(du -sh "$ENGINE_FILE" | cut -f1)"
echo "============================================================"
echo ""
echo "Next step: Update deepstream/config/config_infer_primary.txt"
echo "  model-engine-file=/models/$(basename "$ENGINE_FILE")"
echo ""
