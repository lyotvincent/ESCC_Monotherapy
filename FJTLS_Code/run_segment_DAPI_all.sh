#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT_DIR}/segment_DAPI.py"

REGISTERED_ROOT="data/registered"
COORD_ROOT="data/patches"
SAVE_ROOT="data/nucleiseg"
LOG_DIR="${SAVE_ROOT}/logs"
SUMMARY="${SAVE_ROOT}/segment_DAPI_batch_summary.tsv"

GPU_IDS="${GPU_IDS:-0,1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CELLPOSE_BATCH_SIZE="${CELLPOSE_BATCH_SIZE:-128}"
NITER="${NITER:-500}"
INFERENCE_SCALE="${INFERENCE_SCALE:-1.0}"

mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [ "${SKIP_GPU_CHECK:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  busy_gpus=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v ids="${GPU_IDS}" '
      BEGIN {
        n = split(ids, parts, ",");
        for (i = 1; i <= n; i++) {
          gsub(/ /, "", parts[i]);
          want[parts[i]] = 1;
        }
      }
      {
        gsub(/ /, "", $1);
        if (($1 in want) && $2 + 0 > 20) {
          printf "%s(%s%%) ", $1, $2 + 0;
        }
      }')
  if [ -n "${busy_gpus}" ]; then
    echo "GPU busy: ${busy_gpus}" >&2
    echo "Wait until requested GPUs are idle, or run with SKIP_GPU_CHECK=1 to override." >&2
    exit 1
  fi
fi

printf "sample_id\tstatus\toutput\tlog\tseconds\n" >"${SUMMARY}"

find "${REGISTERED_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r sample_dir; do
  sample_id=$(basename "${sample_dir}")
  dapi_path="${sample_dir}/registered_slides/DAPI.ome.tiff"
  coord_path="${COORD_ROOT}/${sample_id}.h5"
  output_name="${sample_id}_cellposesam_nuclei.npy"
  output_path="${SAVE_ROOT}/${output_name}"
  log_path="${LOG_DIR}/${sample_id}.log"

  if [ -s "${output_path}" ] && [ "${FORCE:-0}" != "1" ]; then
    printf "%s\tskipped_existing\t%s\t%s\t0\n" "${sample_id}" "${output_path}" "${log_path}" >>"${SUMMARY}"
    continue
  fi
  if [ ! -f "${dapi_path}" ]; then
    printf "%s\tmissing_dapi\t%s\t%s\t0\n" "${sample_id}" "${output_path}" "${log_path}" >>"${SUMMARY}"
    continue
  fi
  if [ ! -f "${coord_path}" ]; then
    printf "%s\tmissing_coords\t%s\t%s\t0\n" "${sample_id}" "${output_path}" "${log_path}" >>"${SUMMARY}"
    continue
  fi

  start_time=$(date +%s)
  echo "[$(date '+%F %T')] START ${sample_id}" | tee "${log_path}"
  if "${PYTHON}" "${SCRIPT}" \
    --path "${dapi_path}" \
    --coords "${coord_path}" \
    --save "${SAVE_ROOT}" \
    --output-name "${output_name}" \
    --format npy \
    --batch-size "${BATCH_SIZE}" \
    --cellpose-batch-size "${CELLPOSE_BATCH_SIZE}" \
    --niter "${NITER}" \
    --inference-scale "${INFERENCE_SCALE}" \
    --gpu-ids "${GPU_IDS}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    >>"${log_path}" 2>&1; then
    end_time=$(date +%s)
    seconds=$((end_time - start_time))
    echo "[$(date '+%F %T')] DONE ${sample_id} ${seconds}s" | tee -a "${log_path}"
    printf "%s\tdone\t%s\t%s\t%s\n" "${sample_id}" "${output_path}" "${log_path}" "${seconds}" >>"${SUMMARY}"
  else
    end_time=$(date +%s)
    seconds=$((end_time - start_time))
    echo "[$(date '+%F %T')] FAILED ${sample_id} ${seconds}s" | tee -a "${log_path}"
    printf "%s\tfailed\t%s\t%s\t%s\n" "${sample_id}" "${output_path}" "${log_path}" "${seconds}" >>"${SUMMARY}"
    exit 1
  fi
done
