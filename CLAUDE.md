# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## What this repo is

POC benchmark cho dự án CCTV tại **Meiko Hòa Bình**: so sánh hiệu năng pipeline **DeepStream/TensorRT** vs **baseline OpenCV+PyTorch** khi chạy inference YOLO (FPS/latency/CPU/GPU). Model dùng để benchmark (`models/best.pt`, fine-tune trên dữ liệu PCB) — **model subject không quan trọng**, mục tiêu là so sánh metric pipeline, không phải độ chính xác detection.

## Layout

```
deepstream/    # Dockerfile + config DeepStream, deepstream_benchmark.py
src/           # baseline_benchmark.py, benchmark_multicam*.py, script demo
scripts/       # export_to_trt.sh — export .pt -> ONNX -> TensorRT engine
models/        # best.pt, yolo11n.pt — KHÔNG commit, đã có trong .gitignore
results/       # *.csv metrics output — có thể quy định lại theo Makefile
```

Quy trình chuẩn nằm trong `Makefile` (comment đầu file ghi rõ workflow: `inspect` → `build-ds` → `export` → `baseline`/`deepstream` → `report`). Đọc `Makefile` trước khi chạy tay từng script.

## Ràng buộc

- `models/*.pt`, `*.onnx`, `*.engine` đã nằm trong `.gitignore` — không dùng `git add -f` để commit đè lên (2 file `best.pt`/`yolo11n.pt` từng bị commit nhầm trước khi có rule, đã untrack).
- Benchmark DeepStream cần Docker + GPU NVIDIA thật, không chạy được trong môi trường agent — chỉ kiểm tra syntax/logic script.
- `src/` có nhiều file backup/nháp (`benchmark_multicam_backup.py`, `a.py`, `Code.ipynb`) — không rõ file nào là "chính thức", hỏi lại người dùng trước khi xóa thay vì tự ý dọn.
