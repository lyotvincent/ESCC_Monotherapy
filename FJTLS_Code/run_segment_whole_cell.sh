#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 <sample_id>" >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

SAMPLE_ID="$1"
NUCLEI_DIR="data/nucleiseg"
OUTPUT_DIR="data/wholecellseg"

mkdir -p "${OUTPUT_DIR}"

python segment_whole_cell.py \
  --sample_id "${SAMPLE_ID}" \
  --nuclei_dir "${NUCLEI_DIR}" \
  --output_dir "${OUTPUT_DIR}"
