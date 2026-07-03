#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"

usage() {
  cat <<'EOF'
Usage:
  bash setup_conda_and_export_onnx.sh [env_name] [options]

Options:
  --env-name <name>         Conda environment name (default: yoso)
  --checkpoint <path>       Checkpoint path (default: models/yoso_res50_coco.pth)
  --output <path>           ONNX output path (default: models/yoso_res50.onnx)
  --height <int>            Static export height (default: 512)
  --width <int>             Static export width (default: 512)
  --opset <int>             ONNX opset version (default: 18)
  --dcn-mode <mode>         DCN export mode: native|custom|gridsampl (default: native)
  --legacy-export           Use approximate model_arch export (not recommended)
  --dynamic                 Export dynamic-shape ONNX
  --simplify                Enable onnxsim simplification (default: enabled)
  --no-simplify             Disable onnxsim simplification
  --skip-conda              Do not use conda; use current Python environment
  -h, --help                Show this help

Notes:
  - Positional env_name is kept for backward compatibility.
  - When --dynamic is used, --height/--width are ignored.
  - You can override executables with PYTHON_BIN and PIP_BIN.
EOF
}

ENV_NAME="yoso_cpp"
CHECKPOINT="models/yoso_res50_coco.pth"
OUTPUT="models/yoso_res50.onnx"
HEIGHT="480"
WIDTH="640"
OPSET="18"
DCN_MODE="native"
LEGACY_EXPORT=0
DYNAMIC=0
SIMPLIFY=1
SKIP_CONDA=0

# Backward-compatible positional env name
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  ENV_NAME="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --opset)
      OPSET="$2"
      shift 2
      ;;
    --dcn-mode)
      DCN_MODE="$2"
      shift 2
      ;;
    --legacy-export)
      LEGACY_EXPORT=1
      shift
      ;;
    --dynamic)
      DYNAMIC=1
      shift
      ;;
    --simplify)
      SIMPLIFY=1
      shift
      ;;
    --no-simplify)
      SIMPLIFY=0
      shift
      ;;
    --skip-conda)
      SKIP_CONDA=1
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

if [[ "${DCN_MODE}" != "native" && "${DCN_MODE}" != "custom" && "${DCN_MODE}" != "gridsampl" ]]; then
  echo "Error: --dcn-mode must be 'native', 'custom', or 'gridsampl'"
  exit 1
fi

if [[ "${SKIP_CONDA}" -eq 0 ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"

    # Conda returns non-zero when env does not exist.
    if ! conda run -n "${ENV_NAME}" python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
      conda create -y -n "${ENV_NAME}" python=3.10
    fi

    conda activate "${ENV_NAME}"
  else
    echo "[WARN] conda is not installed or not in PATH."
    echo "[WARN] Continuing with current environment using ${PYTHON_BIN}/${PIP_BIN}."
  fi
fi

# Step 1 – Python requirements
"${PIP_BIN}" install torch torchvision
"${PIP_BIN}" install onnx onnxruntime
"${PIP_BIN}" install onnxsim
"${PIP_BIN}" install onnxruntime-gpu

# Step 2 – Export to ONNX
if [[ "${LEGACY_EXPORT}" -eq 1 ]]; then
  echo "[WARN] Using legacy approximate export (model_arch.py)."
  EXPORT_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convertion/export_to_onnx.py"
    --checkpoint "${CHECKPOINT}"
    --output "${OUTPUT}"
    --opset "${OPSET}"
    --dcn-mode "${DCN_MODE}"
  )
  if [[ "${DYNAMIC}" -eq 1 ]]; then
    EXPORT_CMD+=(--dynamic)
  else
    EXPORT_CMD+=(--height "${HEIGHT}" --width "${WIDTH}")
  fi
else
  echo "[INFO] Using official Detectron2 YOSO export."
  EXPORT_CMD=(
    env PYTHONPATH="${SCRIPT_DIR}/vendor/yoso:${SCRIPT_DIR}/vendor/yoso/projects/YOSO:${PYTHONPATH:-}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/convertion/export_yoso_detectron2_onnx.py"
    --checkpoint "${CHECKPOINT}"
    --output "${OUTPUT}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --opset "${OPSET}"
    --dcn-mode gridsampl
    --device cpu
  )
fi

if [[ "${SIMPLIFY}" -eq 1 ]]; then
  EXPORT_CMD+=(--simplify)
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${EXPORT_CMD[@]}"
