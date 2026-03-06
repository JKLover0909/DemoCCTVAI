# =============================================================================
# deepstream/Dockerfile
#
# DeepStream 8.0 container with:
#   - YOLO11-compatible custom nvinfer parser (via DeepStream-Yolo)
#   - Python benchmark dependencies (pynvml, psutil, numpy)
#
# Build: docker build -t deepstream-ppe:latest deepstream/
# Run:   bash deepstream/run_benchmark.sh
#
# PRE-REQUISITE — login to NGC first (free account required):
#   docker login nvcr.io -u '$oauthtoken' --password-stdin <<< "<YOUR_NGC_API_KEY>"
#   API key: https://ngc.nvidia.com/setup/api-key
# =============================================================================
FROM nvcr.io/nvidia/deepstream:8.0-samples-multiarch

# Use bash so && chains and [ ] conditionals behave predictably
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    g++ \
    python3-pip \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    cuda-cudart-dev-12-6 \
    cuda-nvcc-12-6 \
    libcublas-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python packages
# (numpy already in base image; pynvml + psutil added for metrics)
# --break-system-packages required on Python 3.12+ (PEP 668)
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --break-system-packages pynvml psutil numpy

# ---------------------------------------------------------------------------
# Build DeepStream-Yolo custom parser
#   Supports YOLO5 / YOLO8 / YOLO11 output format parsing for nvinfer.
#   Source: https://github.com/marcoslucianops/DeepStream-Yolo
#
#   CUDA_VER is hardcoded to 12.6 — DeepStream 8.0 always ships with CUDA 12.6.
#   (Auto-detection via nvcc/version.json is unreliable in /bin/sh build context)
# ---------------------------------------------------------------------------
ENV CUDA_VER=12.6

RUN git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git /opt/DeepStream-Yolo
RUN echo "Building DeepStream-Yolo with CUDA_VER=${CUDA_VER}" && \
    make CUDA_VER=${CUDA_VER} -C /opt/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo

# Verify the .so was built
RUN ls -lh /opt/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so

# ---------------------------------------------------------------------------
# Set up application directory
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy DeepStream benchmark files into the image
# (configs, benchmark script, labels)
#   models/   → mounted at runtime via -v
#   results/  → mounted at runtime via -v
COPY . /app/

# Ensure results directory exists inside image (will be overridden by mount)
RUN mkdir -p /results

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV GST_DEBUG=2
ENV DISPLAY=:0
# pyds (DeepStream Python bindings) lives in the DS lib directory
ENV PYTHONPATH=/opt/nvidia/deepstream/deepstream/lib:$PYTHONPATH

# Default command — can be overridden via run_benchmark.sh
CMD ["python3", "/app/deepstream_benchmark.py", "--help"]
