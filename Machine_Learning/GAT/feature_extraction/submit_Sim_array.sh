#!/bin/bash
#SBATCH -J SimG4Test
#SBATCH -o /public/home/liuz1/work/26.03.18_rest/test_example/log/job_SimG4.%A_%a.out
#SBATCH -e /public/home/liuz1/work/26.03.18_rest/test_example/log/job_SimG4.%A_%a.err
#SBATCH --partition=cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load bash environment and rest environment.
source ~/.bashrc
source ~/env_rest.sh

# Make sure this is running as part of a SLURM job array.
ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${ID}" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is empty. Submit with --array=..."
    exit 1
fi

# XXX should be 1..200.
# We rely on SLURM array indexing directly: ID = 1..200.
XXX="${ID}"

WORKDIR="/public/home/liuz1/work/26.03.18_rest/test_example"

INPUT_FILE="${WORKDIR}/G4Events/N500/original_gamma_N500_${XXX}.root"
OUTPUT_FILE="${WORKDIR}/rebuildEvents/Xe_sim_${XXX}.root"


cd ${WORKDIR}/simlulation


unset DISPLAY
export QT_QPA_PLATFORM=offscreen
export ROOTSYS=/public/home/liuz1/apps/root-6.20.00/install
export GROOT_Batch=1
echo "DISPLAY=${DISPLAY-<unset>} QT_QPA_PLATFORM=${QT_QPA_PLATFORM-<unset>} GROOT_Batch=${GROOT_Batch-<unset>}"


restManager \
  -c IonSimu.rml \
  -i "${INPUT_FILE}" \
  -o "${OUTPUT_FILE}" \
  -e 500

# sbatch --array=1-200%20 submit_Sim_array.sh
# scontrol show job xx
