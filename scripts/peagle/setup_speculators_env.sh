#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPECULATORS_ROOT="${SPECULATORS_ROOT:-${PROJECT_DIR}/third_party/speculators}"
SPECULATORS_VENV="${SPECULATORS_VENV:-${PROJECT_DIR}/.speculators_venv}"

cd "${PROJECT_DIR}"

if [[ ! -d "${SPECULATORS_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${SPECULATORS_ROOT}")"
  git clone https://github.com/vllm-project/speculators.git "${SPECULATORS_ROOT}"
else
  git -C "${SPECULATORS_ROOT}" fetch --all --tags --prune
fi

python3 scripts/peagle/patch_speculators_checkout.py \
  --speculators-root "${SPECULATORS_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv or create ${SPECULATORS_VENV} manually." >&2
  exit 1
fi

uv venv "${SPECULATORS_VENV}"
source "${SPECULATORS_VENV}/bin/activate"
uv pip install -r requirements-peagle.txt
uv pip install -e "${SPECULATORS_ROOT}"

export PYTHONPATH="${SPECULATORS_ROOT}/src:${SPECULATORS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/peagle/check_vllm_speculative_support.py
python - <<'PY'
import importlib.util
import speculators

print(f"speculators package: {speculators.__file__}")
missing = [
    name
    for name in ("speculators.models.metrics", "speculators.models.peagle")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(f"missing required speculators modules: {missing}")
PY

echo "Speculators root: ${SPECULATORS_ROOT}"
echo "Speculators venv: ${SPECULATORS_VENV}"
