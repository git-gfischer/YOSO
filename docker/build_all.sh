#!/usr/bin/env bash
# Full YOSO artifact pipeline for Docker image builds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${YOSO_CHECKPOINT:-models/yoso_res50_coco.pth}"
ONNX_OUT="${YOSO_ONNX:-models/yoso_res50.onnx}"
ENGINE_OUT="${YOSO_ENGINE:-models/yoso_res50.engine}"
HEIGHT="${YOSO_HEIGHT:-480}"
WIDTH="${YOSO_WIDTH:-640}"
OPSET="${YOSO_OPSET:-18}"
BUILD_ENGINE="${BUILD_ENGINE:-1}"
BUILD_CPP="${BUILD_CPP:-1}"
INSTALL_CPP_TO="${INSTALL_CPP_TO:-/usr/local/bin}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[build_all][ERROR] Checkpoint not found: ${CHECKPOINT}"
  echo "  Place yoso_res50_coco.pth under models/ before building the image."
  exit 1
fi

echo "=== [1/3] Export ONNX (real Detectron2 YOSO, ${HEIGHT}x${WIDTH}) ==="
python3 convertion/export_yoso_detectron2_onnx.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${ONNX_OUT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --opset "${OPSET}" \
  --dcn-mode gridsampl \
  --device cpu

if [[ "${BUILD_ENGINE}" == "1" ]]; then
  echo "=== [2/3] Build TensorRT engine ==="
  if bash build_tensorrt_engines.sh \
      --skip-deps \
      --mode fp32 \
      --onnx "${ONNX_OUT}" \
      --engine "${ENGINE_OUT}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}"; then
    echo "    Engine built: ${ENGINE_OUT}"
  else
    echo "[build_all][WARN] TRT engine build failed (no GPU during docker build?)."
    echo "  Engine will be built automatically on first 'docker compose up' (GPU required)."
  fi
else
  echo "=== [2/3] Skipping TensorRT engine in image build (BUILD_ENGINE=0) ==="
  echo "  Engine is built on first container start via docker/entrypoint.sh"
fi

if [[ "${BUILD_CPP}" == "1" ]]; then
  echo "=== [3/3] Build C++ inference binaries ==="
  bash cpp/build.sh
  if [[ -n "${INSTALL_CPP_TO}" ]]; then
    install -m 0755 cpp/build/trt/yoso_infer_trt "${INSTALL_CPP_TO}/yoso_infer_trt"
    install -m 0755 cpp/build/onnx/yoso_infer_ort "${INSTALL_CPP_TO}/yoso_infer_ort"
  fi
else
  echo "=== [3/3] Skipping C++ build (BUILD_CPP=0) ==="
fi

echo "[build_all] Done."
echo "  ONNX   : ${ONNX_OUT}"
[[ "${BUILD_ENGINE}" == "1" ]] && echo "  Engine : ${ENGINE_OUT}"
[[ "${BUILD_CPP}" == "1" ]] && echo "  TRT bin: cpp/build/trt/yoso_infer_trt"
[[ "${BUILD_CPP}" == "1" ]] && echo "  ORT bin: cpp/build/onnx/yoso_infer_ort"
