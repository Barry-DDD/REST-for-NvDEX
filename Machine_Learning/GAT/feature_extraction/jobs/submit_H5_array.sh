#!/bin/bash
#SBATCH -J RootToH5
#SBATCH -o /public/home/liuz1/work/26.03.18_rest/test_example/log/job_RootToH5.%A_%a.out
#SBATCH -e /public/home/liuz1/work/26.03.18_rest/test_example/log/job_RootToH5.%A_%a.err
#SBATCH --partition=cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

module load miniconda3/22.11.1-gcc-10.3.0
source /public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-22.11.1-jrusysshr36o4fjbexlhrpljufoxhu4b/etc/profile.d/conda.sh

conda activate pyg-env

ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${ID}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is empty. Submit with --array=..."
    exit 1
fi

WORKDIR="/public/home/liuz1/work/26.03.18_rest/test_example"
SCRIPT_DIR="${WORKDIR}/feature_extraction"

#INPUT_FILE="${WORKDIR}/rebuildEvents/elecsim_12mm_normalRoot/processed_Xe_sim_${ID}.root"
#OUTPUT_DIR="${WORKDIR}/rebuildEvents/elecsim_12mm_h5"
#OUTPUT_FILE="${OUTPUT_DIR}/processed_Xe_sim_${ID}.h5"
#
INPUT_FILE="${WORKDIR}/rebuildEvents/normal_root_detsim_6mm/processed_Xe_sim_${ID}_6mm.root"
OUTPUT_DIR="${WORKDIR}/rebuildEvents/detsim_6mm_h5"
OUTPUT_FILE="${OUTPUT_DIR}/processed_Xe_sim_${ID}.h5"


if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"

python3 "${SCRIPT_DIR}/root_to_hdf5.py" "${INPUT_FILE}" "${OUTPUT_FILE}" --tree HitTree

# Example:
# sbatch --array=1-200%20 submit_H5_array.sh
