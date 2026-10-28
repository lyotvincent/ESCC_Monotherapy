#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

NUCLEI_DIR="data/nucleiseg"
OUTPUT_DIR="data/cell_marker_matrix"

if ! ls "${NUCLEI_DIR}"/*.npy >/dev/null 2>&1; then
  echo "No *.npy files found in ${NUCLEI_DIR}" >&2
  exit 1
fi

for nuclei_path in "${NUCLEI_DIR}"/*.npy; do
  filename="$(basename "${nuclei_path}")"
  sample_id="${filename%.npy}"
  output_path="${OUTPUT_DIR}/${sample_id}/${sample_id}_cell_marker_matrix.h5ad"

  if [ -f "${output_path}" ]; then
    echo "[skip] ${sample_id}: ${output_path} exists"
    continue
  fi

  echo "[run] ${sample_id}"
  sh ./run_generate_cell_marker_matrix.sh "${sample_id}"
done
