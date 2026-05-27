# YOSO ResNet-50 COCO – ONNX / TensorRT Conversion & C++ Inference

Complete pipeline to convert `yoso_res50_coco.pth` into a deployable
ONNX or TensorRT model with C++ inference code.

---

## Architecture summary

| Component  | Details |
|------------|---------|
| Backbone   | ResNet-50 (Detectron2 naming: stem + res2/3/4/5) |
| Neck       | Deformable-Conv FPN (3 DCNv2 stages + location projection) |
| Head       | 100 learned kernels, 2 decoder stages with hash attention + FFN |
| Classes    | 134 (80 COCO things + 53 stuff + 1 background) |
| Parameters | 42.1 M |
| Input      | `[B, 3, H, W]`  H,W must be multiples of 32 |
| Outputs    | `logits [B,100,134]`  ·  `masks [B,100,H/4,W/4]` |

---

## File structure

```
YOSO/
├── convertion/
│   ├── model_arch.py            ← Full PyTorch architecture reconstruction
│   ├── export_to_onnx.py        ← ONNX export script
│   └── build_tensorrt_engine.py ← TensorRT engine builder
├── models/                       ← Checkpoints, ONNX models, TRT engines
├── setup_conda_and_export_onnx.sh ← Conda env + pip install + ONNX export
├── build_tensorrt_engines.sh      ← TensorRT dependency install + engine build
├── cpp/
│   ├── CMakeLists.txt            ← top-level C++ build dispatcher
│   ├── trt/
│   │   ├── CMakeLists.txt        ← TensorRT-only target config
│   │   └── yoso_infer_trt.cpp    ← C++ TensorRT inference
│   └── onnx/
│       ├── CMakeLists.txt        ← ONNX Runtime-only target config
│       └── yoso_infer_ort.cpp    ← C++ ONNX Runtime inference (portable)
├── docker/
│   ├── dockerfile
│   └── docker-compose.yaml
└── README.md
```

# Docker Build
```
docker compose -f docker/docker-compose.yaml build
```

---

## Quick-start scripts

Use the helper scripts at repo root instead of running Step 1/2/3 manually.

### 1) Create conda env + install deps + export ONNX

`setup_conda_and_export_onnx.sh` creates/activates a conda environment,
installs Python dependencies, and runs `convertion/export_to_onnx.py`.

```bash
# Default run (env=yoso_cpp, static 480x640, opset 18, dcn-mode gridsampl)
bash setup_conda_and_export_onnx.sh

# Same, using flags
bash setup_conda_and_export_onnx.sh \
  --env-name my_yoso_env \
  --checkpoint models/yoso_res50_coco.pth \
  --output models/yoso_res50.onnx \
  --height 512 \
  --width 512 \
  --opset 18 \
  --dcn-mode custom \
  --simplify

# Dynamic ONNX export (height/width ignored in dynamic mode)
bash setup_conda_and_export_onnx.sh \
  --env-name my_yoso_env \
  --output models/yoso_res50_dyn.onnx \
  --dcn-mode gridsampl \
  --dynamic
```

Main options:
- `--env-name`
- `--checkpoint`
- `--output`
- `--height`, `--width`
- `--opset`
- `--dcn-mode custom|gridsampl`
- `--dynamic`
- `--simplify` / `--no-simplify`

### 2) Build TensorRT engine (choose one mode)

`build_tensorrt_engines.sh` installs TensorRT Python deps (unless skipped) and
builds exactly one engine mode per run using `convertion/build_tensorrt_engine.py`.

```bash
# FP32 engine (default mode)
bash build_tensorrt_engines.sh
# equivalent:
bash build_tensorrt_engines.sh --mode fp32

# FP16 engine (dynamic profile)
bash build_tensorrt_engines.sh --mode fp16

# FP16 with custom profile/workspace
bash build_tensorrt_engines.sh --mode fp16 \
  --min 1,3,384,384 \
  --opt 1,3,512,512 \
  --max 1,3,768,768 \
  --workspace 4096

# If dependencies are already installed
bash build_tensorrt_engines.sh --mode fp16 --skip-deps
```

Main options:
- `--mode fp32|fp16`
- `--onnx`
- `--engine`
- `--min`, `--opt`, `--max` (FP16)
- `--workspace` (FP16)
- `--skip-deps`

FP32 vs FP16:
- FP32: highest precision, usually slower and higher VRAM use.
- FP16: faster and lower memory on modern NVIDIA GPUs, with possible small
  numeric differences.

**Compatibility notes**
- Prefer `--opset 18` for native DCNv2 support (`torchvision >= 0.13`).
- TensorRT >= 8.6 handles standard ONNX DCNv2 nodes natively.
- For TensorRT 8.4/8.5 install the community DCN plugin:
  <https://github.com/NVIDIA/TensorRT/tree/main/plugin/deformableConvPlugin>
- ONNX Runtime validation requires `--dcn-mode gridsampl` unless a custom DCN plugin/op is installed.

---

## Docker (GPU)

Docker files are in `docker/`:
- `docker/dockerfile`
- `docker/docker-compose.yaml`

Run:

```bash
cd docker
docker compose up --build
```

Open a shell in the running container:

```bash
docker exec -it yoso-cpp bash
```

The Docker image compiles both C++ binaries during build:
- `yoso_infer_trt`
- `yoso_infer_ort`

using:
- `-DBUILD_TRT=ON`
- `-DBUILD_ORT=ON`
- `-DTRT_ROOT=/usr`
- `-DORT_ROOT=/opt/onnxruntime`

---

## C++ inference

### ONNX Runtime (recommended – no CUDA required for CPU)

```bash
# Download ONNX Runtime from:
# https://github.com/microsoft/onnxruntime/releases
# e.g. onnxruntime-linux-x64-1.20.1.tgz  (CPU)
#      onnxruntime-linux-x64-gpu-1.20.1.tgz  (CUDA 12)

cd cpp
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_ORT=ON \
      -DBUILD_TRT=OFF \
      -DORT_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1
cmake --build build -j$(nproc)

# CPU inference
./build/onnx/yoso_infer_ort --model  ../models/yoso_res50.onnx \
                 --image  /path/to/image.jpg    \
                 --height 512 --width 512

# GPU inference (CUDA EP)
./build/onnx/yoso_infer_ort --model  ../models/yoso_res50.onnx \
                 --image  /path/to/image.jpg    \
                 --gpu

# Webcam (device 0)
./build/onnx/yoso_infer_ort --model ../models/yoso_res50.onnx \
                 --webcam --camera-id 0 --gpu
```

### TensorRT

```bash
cd cpp
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TRT=ON \
      -DBUILD_ORT=OFF \
      -DTRT_ROOT=/usr
cmake --build build -j$(nproc)

./build/trt/yoso_infer_trt --engine ../models/yoso_res50.engine \
                 --image  /path/to/image.jpg        \
                 --score-thresh 0.5                 \
                 --mask-thresh  0.5                 \
                 --out-dir      ./results

# Webcam (device 0)
./build/trt/yoso_infer_trt --engine ../models/yoso_res50.engine \
                 --webcam --camera-id 0

# Debug model outputs (top predictions + mask stats per frame)
./build/trt/yoso_infer_trt --engine ../models/yoso_res50.engine \
                 --webcam --camera-id 0 --debug-preds
```

### Build both targets in one configure

```bash
cd cpp
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TRT=ON \
      -DBUILD_ORT=ON \
      -DTRT_ROOT=/usr \
      -DORT_ROOT=/opt/onnxruntime
cmake --build build -j$(nproc)
```

---

## Inference output format

Both executables:
1. Print detected instances with class name and confidence score.
2. Save a visualised image with coloured mask overlays and bounding boxes
   to `--out-dir` (default `./results`).

---

## Performance guide

| Runtime | Precision | 512×512 (A100) | Notes |
|---------|-----------|----------------|-------|
| PyTorch | FP32 | ~35 ms | baseline |
| ONNX Runtime (CUDA EP) | FP32 | ~22 ms | easy deployment |
| TensorRT | FP32 | ~18 ms | best latency |
| TensorRT | FP16 | ~10 ms | Ampere/Ada/Hopper |

---

## Normalisation (must match training)

```
mean = [123.675, 116.280, 103.530]   # Detectron2 RGB pixel mean
std  = [58.395, 57.120, 57.375]      # Detectron2 RGB pixel std
pixel = (pixel_rgb - mean) / std      # pixel in 0..255 scale
```

Input layout: **CHW**, **RGB**, batch-first: `[B, 3, H, W]`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch.onnx.export` fails on DCN | Ensure `torchvision ≥ 0.13` and `--opset 18` |
| TRT engine build fails on DCN node | Upgrade to TRT ≥ 8.6 or install DCN plugin |
| Numeric mismatch > 1e-3 | Check for non-exportable custom ops; review `_register_dcn_symbolic` |
| ORT CUDA EP not found | Install `onnxruntime-gpu` and matching CUDA/cuDNN |
| Empty detections | Lower `--score-thresh`; verify Detectron2 normalization and RGB channel order |
