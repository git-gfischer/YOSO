#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"

usage() {
  cat <<'EOF'
Usage:
  bash build_tensorrt_engines.sh [options]

Options:
  --mode <fp32|fp16>     Precision/mode to build (default: fp32)
  --onnx <path>          ONNX input path
                         fp32 default: models/yoso_res50.onnx
                         fp16 default: models/yoso_res50_dyn.onnx
  --engine <path>        Engine output path
                         fp32 default: models/yoso_res50.engine
                         fp16 default: models/yoso_res50_fp16.engine
  --min <shape>          FP16 min shape (default: 1,3,384,384)
  --opt <shape>          FP16 opt shape (default: 1,3,512,512)
  --max <shape>          FP16 max shape (default: 1,3,768,768)
  --workspace <mb>       FP16 workspace in MB (default: 4096)
  --skip-deps            Skip 'pip install tensorrt pycuda'
  -h, --help             Show this help

Environment overrides:
  PYTHON_BIN             Python executable (default: python3)
  PIP_BIN                Pip executable (default: pip3)

FP32 vs FP16:
  FP32:
    - Highest numerical precision
    - Usually slower and uses more VRAM
    - Good baseline for debugging/validation
    - This script builds FP32 in static-shape mode
  FP16:
    - Lower precision (half-precision float)
    - Typically faster and lower memory on modern NVIDIA GPUs
    - Small accuracy differences can happen in some models
EOF
}

MODE="fp32"
ONNX_FP32="models/yoso_res50.onnx"
ENGINE_FP32="models/yoso_res50.engine"
ONNX_FP16="models/yoso_res50_dyn.onnx"
ENGINE_FP16="models/yoso_res50_fp16.engine"
MIN_SHAPE="1,3,384,384"
OPT_SHAPE="1,3,512,512"
MAX_SHAPE="1,3,768,768"
WORKSPACE="4096"
INSTALL_DEPS=1
CUSTOM_ONNX=""
CUSTOM_ENGINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --onnx)
      CUSTOM_ONNX="$2"
      shift 2
      ;;
    --engine)
      CUSTOM_ENGINE="$2"
      shift 2
      ;;
    --min)
      MIN_SHAPE="$2"
      shift 2
      ;;
    --opt)
      OPT_SHAPE="$2"
      shift 2
      ;;
    --max)
      MAX_SHAPE="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --skip-deps)
      INSTALL_DEPS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "fp32" && "${MODE}" != "fp16" ]]; then
  echo "Error: --mode must be 'fp32' or 'fp16'"
  exit 1
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  "${PIP_BIN}" install tensorrt pycuda
fi

if [[ "${MODE}" == "fp32" ]]; then
  ONNX_PATH="${CUSTOM_ONNX:-${ONNX_FP32}}"
  ENGINE_PATH="${CUSTOM_ENGINE:-${ENGINE_FP32}}"
  mkdir -p "$(dirname "${ENGINE_PATH}")"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/convertion/build_tensorrt_engine.py" \
      --onnx   "${ONNX_PATH}" \
      --engine "${ENGINE_PATH}" \
      --static
else
  ONNX_PATH="${CUSTOM_ONNX:-${ONNX_FP16}}"
  ENGINE_PATH="${CUSTOM_ENGINE:-${ENGINE_FP16}}"
  mkdir -p "$(dirname "${ENGINE_PATH}")"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/convertion/build_tensorrt_engine.py" \
      --onnx   "${ONNX_PATH}" \
      --engine "${ENGINE_PATH}" \
      --fp16 \
      --min  "${MIN_SHAPE}" \
      --opt  "${OPT_SHAPE}" \
      --max  "${MAX_SHAPE}" \
      --workspace "${WORKSPACE}"
fi
