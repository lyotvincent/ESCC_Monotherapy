#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

NUCLEI_DIR="data/nucleiseg"
OUTPUT_DIR="data/wholecellseg"

mkdir -p "${OUTPUT_DIR}"

if ! ls "${NUCLEI_DIR}"/*.npy >/dev/null 2>&1; then
  echo "No *.npy files found in ${NUCLEI_DIR}" >&2
  exit 1
fi

for nuclei_path in "${NUCLEI_DIR}"/*.npy; do
  filename="$(basename "${nuclei_path}")"
  sample_id="${filename%.npy}"
  output_path="${OUTPUT_DIR}/${sample_id}.npy"

  if [ -f "${output_path}" ]; then
    echo "[skip] ${sample_id}: ${output_path} exists"
    continue
  fi

  echo "[run] ${sample_id}"
  sh ./run_segment_whole_cell.sh "${sample_id}"
done
