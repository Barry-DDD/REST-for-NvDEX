#!/bin/bash
#SBATCH -J ExtractHits
#SBATCH -o /public/home/liuz1/work/26.03.18_rest/test_example/log/job_Extract.%A_%a.out
#SBATCH -e /public/home/liuz1/work/26.03.18_rest/test_example/log/job_Extract.%A_%a.err
#SBATCH --partition=cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# -----------------------------------------------------------------------------
# Parameterized SLURM array submit script for fine-hit -> readout extraction.
#
# Selects the readout pitch (in mm) at submit time and forwards it to the
# generalized macro extract_detsim_fine_hits_to_readout.C. The output directory and
# filename are tagged with the pitch so different granularities never collide.
#
# Usage:
#   # 1) export the pitch and submit:
#   READOUT_PITCH_MM=8 sbatch --array=1-200%20 submit_Extract_array_readout.sh
#
#   # 2) or pass it through SLURM's --export:
#   sbatch --export=ALL,READOUT_PITCH_MM=3 --array=1-200%20 \
#          submit_Extract_array_readout.sh
#
#   # 3) or pass as a positional argument after the script name:
#   sbatch --array=1-200%20 submit_Extract_array_readout.sh 12
#
# Valid pitches are any positive number; typical values are 2, 3, 8, 12.
# -----------------------------------------------------------------------------

source ~/.bashrc
source ~/env_rest.sh

ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${ID}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is empty. Submit with --array=..."
    exit 1
fi

# Resolve readout pitch in mm.
# Priority: positional argument > environment variable > default 3.
PITCH_INPUT="${1:-${READOUT_PITCH_MM:-3}}"

# Validate: positive number.
if ! awk -v p="${PITCH_INPUT}" 'BEGIN{ exit !(p+0 > 0) }'; then
    echo "Error: readout pitch must be a positive number, got '${PITCH_INPUT}'."
    exit 1
fi

# Canonical numeric form (e.g. "8" or "2.5"), and a filename-friendly tag.
PITCH_MM=$(awk -v p="${PITCH_INPUT}" 'BEGIN{ printf("%g", p+0) }')
PITCH_TAG="${PITCH_MM//./p}mm"   # 2 -> 2mm, 2.5 -> 2p5mm

WORKDIR="/public/home/liuz1/work/26.03.18_rest/test_example"
SCRIPT_DIR="${WORKDIR}/feature_extraction"

# Default input/output layout: detsim inputs, pitch-tagged output directory.
# Override by exporting INPUT_DIR / OUTPUT_DIR / FILE_PREFIX before sbatch.
INPUT_DIR="${INPUT_DIR:-${WORKDIR}/rebuildEvents/detsim}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/rebuildEvents/normal_root_detsim_${PITCH_TAG}}"
FILE_PREFIX="${FILE_PREFIX:-Xe_sim}"

INPUT_FILE="${INPUT_DIR}/${FILE_PREFIX}_${ID}.root"
OUTPUT_FILE="${OUTPUT_DIR}/processed_${FILE_PREFIX}_${ID}_${PITCH_TAG}.root"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

unset DISPLAY
export QT_QPA_PLATFORM=offscreen
export ROOTSYS=/public/home/liuz1/apps/root-6.20.00/install
export GROOT_Batch=1
echo "DISPLAY=${DISPLAY-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM-<unset>} GROOT_Batch=${GROOT_Batch-<unset>}"

echo "Readout pitch: ${PITCH_MM} mm (tag=${PITCH_TAG})"
echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"

restRoot -b -q "${SCRIPT_DIR}/extract_detsim_fine_hits_to_readout.C(\"${INPUT_FILE}\",\"${OUTPUT_FILE}\",${PITCH_MM})"

# Examples:
#   READOUT_PITCH_MM=3  sbatch --array=1-200%20 submit_Extract_detsim.sh 
#   READOUT_PITCH_MM=8  sbatch --array=1-200%20 submit_Extract_detsim.sh 
#   READOUT_PITCH_MM=10  sbatch --array=1-200%20 submit_Extract_detsim.sh 
#   READOUT_PITCH_MM=12 sbatch --array=1-200%20 submit_Extract_detsim.sh 
