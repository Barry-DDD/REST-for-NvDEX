#!/bin/bash
#SBATCH --job-name=model_pid
# #SBATCH --partition=junogpu
# #SBATCH --qos=junoatmgpu
# #SBATCH --account=junogpu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=junogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5        # 1 main + 4 DataLoader workers
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/logs/pid_%j.out
#SBATCH --error=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/logs/pid_%j.err

set -euo pipefail

env_name=ml-v100-torch251
#env_name=ml-env-blackwell
echo "+++++Activate environment"
source /cvmfs/juno.ihep.ac.cn/sw/mamba/Mambaforge-2023.05-Linux-x86_64/etc/profile.d/conda.sh
source /cvmfs/juno.ihep.ac.cn/sw/mamba/Mambaforge-2023.05-Linux-x86_64/etc/profile.d/mamba.sh
conda activate ${env_name}
echo "+++++ done"

#OUTDIR=/lustrefs/juno26/users/liuz3/work/test/GAT/results/pid_2nubb_ex_lr5e-3_wd1e-3_batch64
#INFILE=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/select48K_mixed.h5

OUTDIR=/lustrefs/juno26/users/liuz3/work/test/GAT/results/pid_detsim_2nubb_ex_lr5e-3_wd1e-3_batch64
INFILE=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/detsim_select80K_mixed.h5


mkdir -p "${OUTDIR}"

# criterion = nn.SmoothL1Loss()

torchrun --nproc_per_node=1 \
  /lustrefs/juno26/users/liuz3/work/test/GAT/tpc_GAT/run.py \
  --infile    "${INFILE}"  \
  --outdir    "${OUTDIR}"  \
  --task binary_id         \
  --epochs    30           \
  --batch_size       64   \
  --learning_rate    5e-4 \
  --weight_decay     1e-3 \
  --features x y energy log_energy \
  --feature_norm_mode standard      \
  --radius_mm        24    \
  --hidden_channels  128   \
  --num_layers       3     \
  --heads            4     \
  --num_workers      4

# --learning_rate    1e-3 \

# sbatch sub_job_elecsim_default.sh
