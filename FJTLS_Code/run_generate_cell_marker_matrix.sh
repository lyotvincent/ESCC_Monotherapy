#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: sh $0 <sample_id> [extra_args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

SAMPLE_ID="$1"
shift

python generate_cell_marker_matrix.py \
  --sample-id "${SAMPLE_ID}" \
  "$@"
