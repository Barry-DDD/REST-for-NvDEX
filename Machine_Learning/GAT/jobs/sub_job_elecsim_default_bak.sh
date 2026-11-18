#!/bin/bash
#SBATCH --job-name=pid
#SBATCH --partition=dgx2  #dgx2,dgx1,nv100-sug,nv100-ins
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5        # 1 main + 4 DataLoader workers
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/jobs/jobs/logs/pid_%j.out
#SBATCH --error=/public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/jobs/jobs/logs/pid_%j.err

set -euo pipefail

#OUTDIR=/public/home/liuz1/work/26.03.18_rest/test_example/tpc_gat/results/results_r24_elecsim_12mm
#INFILE=/public/home/liuz1/work/26.03.18_rest/test_example/feature_extraction/output/all_events_elecsim_12mm.h5
OUTDIR=/public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/results/pid_2nubb_ex
INFILE=/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/elecsim_normal_root/all_events_elecsim_12mm.h5

cd /public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/jobs
mkdir -p /public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/jobs/logs
mkdir -p "${OUTDIR}"

# Activate conda environment
source /public/soft/linux-centos7-x86_64/gcc-10.3.0/miniconda3-22.11.1-jrusysshr36o4fjbexlhrpljufoxhu4b/etc/profile.d/conda.sh
conda activate pyg-env

echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Python: $(which python)  Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PORT=$(( (SLURM_JOB_ID % 40000) + 10000 ))
MASTER_ADDR=$(hostname)
echo "Using rdzv_endpoint=${MASTER_ADDR}:${PORT}"

torchrun --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${PORT} \ 
  /public/home/liuz1/work/26.03.18_rest/nubb_ex/GAT/tpc_GAT/run.py \
  --infile    "${INFILE}"  \
  --outdir    "${OUTDIR}"  \
  --task binary_id         \
  --epochs    30           \
  --batch_size       32   \
  --learning_rate    1e-3  \
  --weight_decay     1e-3  \
  --features x y energy log_energy \
  --feature_norm_mode standard      \
  --radius_mm        24    \
  --hidden_channels  128   \
  --num_layers       3     \
  --heads            4     \
  --num_workers      4


# sbatch sub_job.sh


