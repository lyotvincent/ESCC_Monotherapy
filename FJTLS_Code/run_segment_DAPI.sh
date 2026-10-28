#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT_DIR}/segment_DAPI.py"

DAPI_PATH="data/registered/21-07010/registered_slides/DAPI.ome.tiff"
COORD_PATH="data/patches/21-07010.h5"
SAVE_DIR="data/nucleiseg"
OUTPUT_NAME="21-07010_cellposesam_nuclei.npy"

export CUDA_VISIBLE_DEVICES=0,1

if [ "${SKIP_GPU_CHECK:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  busy_gpus=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '$1 ~ /^[[:space:]]*[01][[:space:]]*$/ && $2+0 > 20 {gsub(/ /, "", $1); printf "%s(%s%%) ", $1, $2+0}')
  if [ -n "${busy_gpus}" ]; then
    echo "GPU busy: ${busy_gpus}" >&2
    echo "Wait until GPU0 and GPU1 are idle, or run with SKIP_GPU_CHECK=1 to override." >&2
    exit 1
  fi
fi

"${PYTHON}" "${SCRIPT}" \
  --path "${DAPI_PATH}" \
  --coords "${COORD_PATH}" \
  --save "${SAVE_DIR}" \
  --output-name "${OUTPUT_NAME}" \
  --format npy \
  --batch-size 128 \
  --cellpose-batch-size 128 \
  --niter 500 \
  --inference-scale 1.0 \
  --gpu-ids 0,1 \
  --workers-per-gpu 3 \
  "$@"
