#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Defaults match the Docker image layout.
TRT_ROOT="${TRT_ROOT:-/usr}"
ORT_ROOT="${ORT_ROOT:-/opt/onnxruntime}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "[build.sh] Cleaning build directory: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

echo "[build.sh] Configuring CMake (TRT + ORT)"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DBUILD_TRT=ON \
  -DBUILD_ORT=ON \
  -DTRT_ROOT="${TRT_ROOT}" \
  -DORT_ROOT="${ORT_ROOT}"

echo "[build.sh] Building yoso_infer_trt + yoso_infer_ort"
cmake --build "${BUILD_DIR}" --target yoso_infer_trt yoso_infer_ort -j"$(nproc)"

TRT_BIN="${BUILD_DIR}/trt/yoso_infer_trt"
ORT_BIN="${BUILD_DIR}/onnx/yoso_infer_ort"

if [[ ! -x "${TRT_BIN}" || ! -x "${ORT_BIN}" ]]; then
  echo "[build.sh][ERROR] Build finished but binaries are missing."
  echo "  Expected:"
  echo "    ${TRT_BIN}"
  echo "    ${ORT_BIN}"
  exit 1
fi

echo "[build.sh] Done."
echo "  TensorRT binary: ${TRT_BIN}"
echo "  ONNX Runtime binary: ${ORT_BIN}"
