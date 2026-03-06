#!/usr/bin/env bash
# =============================================================================
# deepstream/run_benchmark.sh
#
# Runs the DeepStream benchmark pipeline inside the pre-built Docker container.
# Mounts models/ and results/ from the project root.
#
# Usage:
#   bash deepstream/run_benchmark.sh
#   bash deepstream/run_benchmark.sh --duration 60 --warmup 15
#   RTSP_URL="rtsp://..." bash deepstream/run_benchmark.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DS_IMAGE="deepstream-ppe:latest"

# Runtime parameters (override via env or CLI args forwarded to the script)
RTSP_URL="${RTSP_URL:-rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101}"
DURATION="${DURATION:-120}"
WARMUP="${WARMUP:-30}"

echo ""
echo "============================================================"
echo "  DeepStream 8.0 Benchmark"
echo "  Image   : $DS_IMAGE"
echo "  URL     : $RTSP_URL"
echo "  Warmup  : ${WARMUP}s"
echo "  Duration: ${DURATION}s"
echo "  Models  : $PROJECT_ROOT/models/"
echo "  Results : $PROJECT_ROOT/results/"
echo "============================================================"
echo ""

# Ensure results directory exists on host
mkdir -p "$PROJECT_ROOT/results"

docker run \
  --rm \
  --gpus all \
  --network host \
  --ipc=host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e RTSP_URL="$RTSP_URL" \
  -e PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib \
  -v "${PROJECT_ROOT}/models:/models:ro" \
  -v "${PROJECT_ROOT}/results:/results:rw" \
  -v "${SCRIPT_DIR}/deepstream_benchmark.py:/app/deepstream_benchmark.py:ro" \
  -v "${SCRIPT_DIR}/config:/app/config:ro" \
  -v "${SCRIPT_DIR}/labels.txt:/app/labels.txt:ro" \
  "$DS_IMAGE" \
  python3 /app/deepstream_benchmark.py \
    --url "$RTSP_URL" \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    "$@"

echo ""
echo "✅ DeepStream benchmark complete."
echo "   CSV output: $PROJECT_ROOT/results/deepstream_metrics.csv"
