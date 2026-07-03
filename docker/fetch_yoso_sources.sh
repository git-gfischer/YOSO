#!/usr/bin/env bash
# Ensure detectron2 + YOSO sources exist under vendor/yoso/ (not tracked in git).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOSO_SRC="${ROOT}/vendor/yoso"
YOSO_REPO="${YOSO_GIT_URL:-https://github.com/hujiecpp/YOSO.git}"

if [[ -f "${YOSO_SRC}/setup.py" && -d "${YOSO_SRC}/projects/YOSO" ]]; then
  echo "[fetch_yoso_sources] Using vendor/yoso/ from build context."
  exit 0
fi

echo "[fetch_yoso_sources] vendor/yoso/ missing — cloning ${YOSO_REPO} …"
mkdir -p "${ROOT}/vendor"
rm -rf /tmp/yoso-src-clone
git clone --depth 1 "${YOSO_REPO}" /tmp/yoso-src-clone
rm -rf "${YOSO_SRC}"
mv /tmp/yoso-src-clone "${YOSO_SRC}"

test -f "${YOSO_SRC}/setup.py"
test -d "${YOSO_SRC}/projects/YOSO"
echo "[fetch_yoso_sources] Ready: ${YOSO_SRC}"
