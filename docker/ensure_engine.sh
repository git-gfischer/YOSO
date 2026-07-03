#!/usr/bin/env bash
# Build TRT engine on first container start (requires GPU + driver).
set -euo pipefail

ROOT="/workspace/YOSO"
cd "${ROOT}"

ONNX="${YOSO_ONNX:-models/yoso_res50.onnx}"
ENGINE="${YOSO_ENGINE:-models/yoso_res50.engine}"
HEIGHT="${YOSO_HEIGHT:-480}"
WIDTH="${YOSO_WIDTH:-640}"

if [[ -f "${ENGINE}" ]]; then
  exit 0
fi

if [[ ! -f "${ONNX}" ]]; then
  echo "[entrypoint] ONNX missing (${ONNX}); skip engine build."
  exit 0
fi

echo "[entrypoint] TRT engine not found — building ${ENGINE} (${HEIGHT}x${WIDTH}) …"
if bash build_tensorrt_engines.sh \
    --skip-deps \
    --mode fp32 \
    --onnx "${ONNX}" \
    --engine "${ENGINE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}"; then
  echo "[entrypoint] Engine ready: ${ENGINE}"
else
  echo "[entrypoint][WARN] Engine build failed (GPU/driver required)."
  echo "  Run manually inside the container:"
  echo "    bash build_tensorrt_engines.sh --skip-deps --mode fp32"
fi
