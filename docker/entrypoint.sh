#!/usr/bin/env bash
set -euo pipefail

bash /workspace/YOSO/docker/ensure_engine.sh
exec "$@"
