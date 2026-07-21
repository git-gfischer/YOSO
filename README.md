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
│   ├── model_arch.py                  ← Legacy approximate arch (do not use)
│   ├── export_yoso_detectron2_onnx.py ← Correct export from real YOSO
│   ├── export_to_onnx.py              ← Legacy export (deprecated)
│   └── build_tensorrt_engine.py       ← TensorRT engine builder
├── models/                            ← Checkpoints, ONNX models, TRT engines
├── vendor/yoso/                       ← Upstream YOSO + detectron2 (not in git)
├── docker/
│   ├── dockerfile                     ← Full automated build
│   ├── docker-compose.yaml
│   ├── build_all.sh                   ← ONNX export + C++ build
│   ├── fetch_yoso_sources.sh          ← Clone YOSO if vendor/yoso/ missing
│   ├── entrypoint.sh                  ← Build TRT engine on first start
│   └── ensure_engine.sh
├── cpp/
│   ├── build.sh
│   ├── trt/yoso_infer_trt.cpp
│   └── onnx/yoso_infer_ort.cpp
└── README.md
```

---

## Docker (recommended)

The Docker image handles the full pipeline automatically:

| Step | When | Notes |
|------|------|-------|
| Clone YOSO / detectron2 sources | `docker compose build` | if `vendor/yoso/` is missing from context |
| Install Python + C++ dependencies | `docker compose build` | torch, detectron2, ORT SDK, OpenCV, … |
| ONNX export (`models/yoso_res50.onnx`) | `docker compose build` | real YOSO, 480×640, gridsampl DCN |
| C++ binaries (`yoso_infer_trt`, `yoso_infer_ort`) | `docker compose build` | installed to `/usr/local/bin/` |
| TRT engine (`models/yoso_res50.engine`) | first `docker compose up` | requires GPU; via `entrypoint.sh` |

### Prerequisites

1. Place the checkpoint at `/workspace/YOSO/models/yoso_res50_coco.pth` (the `models/` directory is gitignored).
2. NVIDIA GPU + drivers for inference and TRT engine build.

### Build and run

```bash
docker compose -f docker/docker-compose.yaml build
xhost +
docker compose -f docker/docker-compose.yaml run yoso bash
```

On first start the entrypoint builds `models/yoso_res50.engine` if it is missing.

Optional build args (`docker build --build-arg` or compose `build.args`):

| Arg | Default | Purpose |
|-----|---------|---------|
| `YOSO_HEIGHT` | `480` | Export / engine input height |
| `YOSO_WIDTH` | `640` | Export / engine input width |
| `BUILD_ENGINE` | `0` | Set `1` to try TRT engine during image build (needs GPU) |

---

## Inference

Binaries are on `PATH` inside the container as `yoso_infer_trt` and `yoso_infer_ort`.

### TensorRT (webcam)

```bash
yoso_infer_trt --engine /workspace/YOSO/models/yoso_res50.engine \
               --webcam --camera-id 0

# Debug model outputs (top predictions + mask stats per frame)
yoso_infer_trt --engine /workspace/YOSO/models/yoso_res50.engine \
               --webcam --camera-id 0 --debug-preds
```

### TensorRT (image)

```bash
yoso_infer_trt --engine /workspace/YOSO/models/yoso_res50.engine \
               --image  /path/to/image.jpg \
               --score-thresh 0.3 \
               --mask-thresh  0.5 \
               --out-dir      ./results
```

### ONNX Runtime

```bash
# GPU inference (CUDA EP)
yoso_infer_ort --model /workspace/YOSO/models/yoso_res50.onnx \
               --image /path/to/image.jpg --gpu

# Webcam
yoso_infer_ort --model /workspace/YOSO/models/yoso_res50.onnx \
               --webcam --camera-id 0 --gpu
```

### Output format

Both executables:
1. Print detected instances with class name and confidence score.
2. Save a visualised image with coloured mask overlays and bounding boxes
   to `--out-dir` (default `./results`).

---

## Normalisation (must match training)

```
mean = [123.675, 116.280, 103.530]   # Detectron2 RGB pixel mean
std  = [58.395, 57.120, 57.375]      # Detectron2 RGB pixel std
pixel = (pixel_rgb - mean) / std      # pixel in 0..255 scale
```

Input layout: **CHW**, **RGB**, batch-first: `[B, 3, H, W]`.

---

## Performance guide

| Runtime | Precision | 512×512 (A100) | Notes |
|---------|-----------|----------------|-------|
| PyTorch | FP32 | ~35 ms | baseline |
| ONNX Runtime (CUDA EP) | FP32 | ~22 ms | easy deployment |
| TensorRT | FP32 | ~18 ms | best latency |
| TensorRT | FP16 | ~10 ms | Ampere/Ada/Hopper |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `vendor/yoso` missing during Docker build | Dockerfile clones it via `fetch_yoso_sources.sh`; or run `git clone https://github.com/hujiecpp/YOSO.git vendor/yoso` |
| Checkpoint not found during build | Place `models/yoso_res50_coco.pth` before `docker compose build` |
| TRT engine missing at runtime | Start container with GPU; entrypoint builds it on first run |
| Empty detections / white masks | Rebuild image (uses correct `export_yoso_detectron2_onnx.py`); try `--debug-preds` |
| Class scores ~0.02 (flat) | Engine built from wrong legacy export — rebuild Docker image |
| Saturated masks (mean ~0.98) | Same as above |
| ORT CUDA EP not found | Use the Docker image (ORT GPU SDK pre-installed) |

---

## Manual build (without Docker)

Use these only if you need a custom setup outside Docker:

- `setup_conda_and_export_onnx.sh` — conda env + ONNX export
- `build_tensorrt_engines.sh` — TRT engine build
- `cpp/build.sh` — compile C++ binaries locally
