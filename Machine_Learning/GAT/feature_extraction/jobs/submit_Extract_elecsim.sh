#!/bin/bash
#SBATCH -J ExtractHits
#SBATCH -o /public/home/liuz1/work/26.03.18_rest/test_example/log/job_Extract.%A_%a.out
#SBATCH -e /public/home/liuz1/work/26.03.18_rest/test_example/log/job_Extract.%A_%a.err
#SBATCH --partition=cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

source ~/.bashrc
source ~/env_rest.sh

ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${ID}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is empty. Submit with --array=..."
    exit 1
fi

WORKDIR="/public/home/liuz1/work/26.03.18_rest/test_example"
SCRIPT_DIR="${WORKDIR}/feature_extraction"


INPUT_FILE="${WORKDIR}/rebuildEvents/elecsim_12mm/Xe_sim_${ID}.root"
OUTPUT_FILE="${WORKDIR}/rebuildEvents/elecsim_12mm_normalRoot/processed_Xe_sim_${ID}.root"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}"
    exit 1
fi

mkdir -p "${WORKDIR}/rebuildEvents/normal_root"

unset DISPLAY
export QT_QPA_PLATFORM=offscreen
export ROOTSYS=/public/home/liuz1/apps/root-6.20.00/install
export GROOT_Batch=1
echo "DISPLAY=${DISPLAY-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM-<unset>} GROOT_Batch=${GROOT_Batch-<unset>}"

echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"

restRoot -b -q "${SCRIPT_DIR}/extract_elecsim_hits.C(\"${INPUT_FILE}\",\"${OUTPUT_FILE}\")"

# sbatch --array=1-200%20 submit_Extract_elecsim.sh
