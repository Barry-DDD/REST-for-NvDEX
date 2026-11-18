#!/bin/bash
#SBATCH --job-name=detsim_theta_angle
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=junogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/logs/theta_angle_%j.out
#SBATCH --error=/lustrefs/juno26/users/liuz3/work/test/GAT/jobs/logs/theta_angle_%j.err

set -euo pipefail
PROJECT=/lustrefs/juno26/users/liuz3/work/test/GAT
# Flat-theta sample (theta12 uniform in [0, 180] deg), cut: total_energy > 1000 keV && hit_count > 0.
INFILE=${INFILE:-${PROJECT}/input/detsim_tehta0vbb_noDiffusion_E1000.h5}
OUTDIR=${OUTDIR:-${PROJECT}/results/detsim_theta_angle_E1000v3_${SLURM_JOB_ID:-manual}}

#INFILE=${INFILE:-${PROJECT}/input/detsim_tehta0vbb_noDiffusion_2mm_selected.h5}
#OUTDIR=${OUTDIR:-${PROJECT}/results/detsim_theta_angle_2mm_${SLURM_JOB_ID:-manual}}

source /cvmfs/juno.ihep.ac.cn/sw/mamba/Mambaforge-2023.05-Linux-x86_64/etc/profile.d/conda.sh
source /cvmfs/juno.ihep.ac.cn/sw/mamba/Mambaforge-2023.05-Linux-x86_64/etc/profile.d/mamba.sh
conda activate /hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251
cd "${PROJECT}"
mkdir -p "${OUTDIR}"

torchrun --standalone --nproc_per_node=1 tpc_GAT/run.py \
  --infile "${INFILE}" --outdir "${OUTDIR}" \
  --task regression_theta --target_mode angle \
  --val_fraction 0.20 --seed 42 \
  --epochs 30 --batch_size 64 \
  --learning_rate 3e-4 --weight_decay 5e-5 \
  --features x y z energy log_energy \
  --vertex_centroid \
  --feature_norm_mode standard --radius_mm 2 \
  --xy_abs_max 445 --z_abs_max 850 \
  --hidden_channels 128 --num_layers 3 --heads 4 --num_workers 4

#  --vertex_relative \
python results/plot_theta_result.py --indir "${OUTDIR}"

# Before submission from the project root:
# mkdir -p jobs/logs
# sbatch jobs/sub_job_detsim_theta_angle.sh
