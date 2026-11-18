# GAT for NvDEx

The [nvdex-ml-workflow quick start](skills/nvdex-ml-workflow/README.md) shows how to
check an HDF5 sample, prepare a direct or SLURM training run, and validate and plot
the results for existing GAT tasks.

Task-aware graph learning for NvDEx TPC events using a Graph Attention Network
(GATv2) on xy-plane hits. The original task reconstructs the primary vertex Z
position (`primary_origin_z`); the same graph input pipeline also supports
event-id classification tasks and the 0vbb primary-electron energy asymmetry A.

The pipeline goes from REST simulation ROOT files to a single merged HDF5
training file, and then trains a graph regression model on a GPU cluster.

```
REST simulation (.root)
        |
        |  feature_extraction/extract_*.C    (REST -> simple TTree)
        v
processed ROOT files (HitTree)
        |
        |  feature_extraction/root_to_hdf5.py
        v
per-file HDF5
        |
        |  feature_extraction/merge_hdf5.py
        v
merged all_events.h5
        |
        |  tpc_GAT/run.py        (task-aware GATv2 training)
        v
trained model + evaluation
```

## Repository layout

```
tpc_gat/
├── feature_extraction/        Convert REST ROOT to ML-ready HDF5
│   ├── extract_elecsim_hits.C            Full electronics simulation -> TTree
│   ├── extract_detsim_fine_hits_to_readout.C
│   │                                     Detector-level fine hits rebinned to a
│   │                                     configurable readout pitch -> TTree
│   ├── root_to_hdf5.py                   Per-file TTree -> HDF5
│   ├── merge_hdf5.py                     Merge per-file HDF5 -> one training file
│   ├── HDF5_SCHEMA.md                    Output HDF5 schema reference
│   └── jobs/                             SLURM submit scripts
│
├── tpc_GAT/                   GATv2 model and training pipeline
│   ├── tpc_graph_dataset.py              HDF5 -> PyG graph dataset
│   ├── gat_model.py                      Edge-aware GATv2 backbone + task heads
│   ├── task_spec.py                      Task definitions and id label mappings
│   ├── utils.py                          DDP-safe train / validation loops
│   ├── run.py                            Training entry point (torchrun)
│   ├── event_stats.py                    Pre-training data inspection
│   └── README.md                         Full model / training documentation
│
└── jobs/                      SLURM submit scripts for GPU training
    ├── check_gpu_partitions.sh           Inspect cluster GPU partition usage
    ├── sub_job_elecsim_default.sh        Train on elecsim HDF5
    └── sub_job_detsim_default.sh         Train on detsim HDF5
```

## Environment

The project was developed on an HPC cluster running CentOS 7 (GLIBC 2.17),
with modules provided through `miniconda3` and the GNU toolchain. Two separate
environments are used:

| Stage | Environment |
| --- | --- |
| ROOT extraction (`extract_*.C`) | REST + ROOT 6.20.00 (with Geant4 / Garfield++) |
| HDF5 conversion / merging       | Python + `h5py` + `uproot`, conda env `pyg-env` |
| Training (`tpc_GAT/run.py`)     | Python + PyTorch + PyTorch Geometric, conda env `pyg-env` |

> The `env_rest.sh` and `env_conda.sh` scripts in this directory are
> **examples of the author's site-specific setup** (absolute paths, module
> names, and a pre-built REST install that are not portable). Use them only as
> a reference for which components to load; reproduce the equivalent on your own
> system as described below.

### REST / ROOT environment (Stage 1)

The ROOT extraction macros require a REST installation built against ROOT
6.20.00, Geant4 10.2.3, and Garfield++. On the author's cluster these are loaded
via environment modules and `thisroot.sh` / `geant4.sh` / `thisREST.sh`
(see `env_rest.sh` for the exact layout). After sourcing, `restRoot` must be on
`PATH`.

### Python environment (`pyg-env`, Stages 1.2–2)

A conda environment built around Python 3.11, PyTorch 2.4.0 (CUDA 12.4), and
PyTorch Geometric. Python 3.12 is avoided because no `torch-scatter` wheel is
published for the PyTorch 2.4 build. To reproduce it:

```bash
# Python 3.11 base (3.12 lacks a torch-scatter wheel for the pt24 build)
conda create -n pyg-env -c conda-forge python=3.11 pip -y
conda activate pyg-env

# Scientific stack from conda-forge (no CUDA libraries here)
conda install -c conda-forge -y mamba
mamba install -c conda-forge -y \
    numpy typing_extensions \
    h5py uproot scipy matplotlib=3.8.3 scikit-learn scikit-image \
    pandas seaborn tqdm boost-histogram scienceplots

python -m pip install --upgrade pip

# PyTorch 2.4.0 + CUDA 12.4 (manylinux2014 wheel, works on GLIBC 2.17 / CentOS 7)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# PyG companion libraries (matching manylinux wheels)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric

# Verify
python -c "import torch, torch_geometric; \
print('torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print('pyg:', torch_geometric.__version__)"
```

Then activate it for any Python stage:

```bash
conda activate pyg-env
```

## Stage 1 — Feature extraction (REST ROOT -> HDF5)

### 1.1 REST ROOT -> processed TTree

Two macros are provided, depending on which simulation stage you want as input.

**Full electronics simulation:**

```bash
# REST/ROOT environment must be loaded first (see Environment section)
restRoot -b -q 'feature_extraction/extract_elecsim_hits.C("input.root","processed.root")'
```

**Detector-level fine hits rebinned to a chosen readout pitch (mm):**

```bash
restRoot -b -q 'feature_extraction/extract_detsim_fine_hits_to_readout.C("input.root","processed.root",3.0)'
```

The pitch (last argument) controls the readout cell size. Typical values are
2, 3, 6, 8, 12 mm. The module footprint is kept fixed; only the local cell
grid is rescaled.

Both macros produce a flat `HitTree` with hit-level and event-level branches.

### 1.2 Processed TTree -> HDF5

```bash
conda activate pyg-env
python feature_extraction/root_to_hdf5.py processed.root processed.h5 --tree HitTree
```

The output HDF5 contains `events/`, `hits/`, `primaries/`, `stats/`, `summary/`
groups. Hits with `energy <= 0` are filtered before writing. See
[`feature_extraction/HDF5_SCHEMA.md`](feature_extraction/HDF5_SCHEMA.md) for the
full schema.

### 1.3 Merge per-file HDF5 -> single training file

```bash
python feature_extraction/merge_hdf5.py all_events.h5 file1.h5 file2.h5 ...
```

Event offsets are re-stitched and normalization statistics are recomputed over
all merged events. The resulting `all_events.h5` is what the training pipeline
consumes.

### 1.4 Cluster submission

SLURM array submit scripts are provided in `feature_extraction/jobs/`. Usage
instructions are at the bottom of each script. Examples:

```bash
# Detsim extraction at a chosen readout pitch
READOUT_PITCH_MM=12 sbatch --array=1-200%20 feature_extraction/jobs/submit_Extract_detsim.sh

# Elecsim extraction
sbatch --array=1-200%20 feature_extraction/jobs/submit_Extract_elecsim.sh

# Processed ROOT -> HDF5
sbatch --array=1-200%20 feature_extraction/jobs/submit_H5_array.sh
```

## Stage 2 — Training the GAT model

The model implements an edge-aware GATv2 network that takes per-event hit graphs
and predicts either the scaled `primary_origin_z` or event classification labels.
See
[`tpc_GAT/README.md`](tpc_GAT/README.md) for the complete model documentation,
dataset fields, hyperparameters, and output file formats.

### Quick start

```bash
conda activate pyg-env

torchrun --nproc_per_node=1 tpc_GAT/run.py \
  --infile  /path/to/all_events.h5 \
  --outdir  tpc_GAT/results/run_demo \
  --task regression_z \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --features x y energy log_energy \
  --feature_norm_mode standard \
  --radius_mm 24 \
  --hidden_channels 128 \
  --num_layers 3 \
  --heads 4
```

To also feed the hit `z` coordinate as an input, add `z` to `--features`. Like
`x/y`, `z` uses fixed symmetric scaling (mean assumed 0), differing only in its
half-range, controlled by `--z_abs_max` (default `850` mm):

```bash
  --features x y z energy log_energy \
  --xy_abs_max 445 --z_abs_max 850 \
```

For binary event classification on files containing `/events/id`, use
`--task binary_id`. The current mapping is `id=1 -> signal label 1` and
`id=21/22/23/24 -> background label 0`.

### Inspect data before training

```bash
python tpc_GAT/event_stats.py /path/to/all_events.h5 \
       --n-events 1000 --radii-mm 15,20,25
```

This reports hit counts, nearest-neighbor distances, candidate edge counts at
each radius, and the energy and z distributions.

### Outputs

After training (`--outdir`):

| File | Description |
| --- | --- |
| `best_model.pt`         | Model weights at the lowest validation loss epoch. |
| `best_evaluation.npz`   | Task predictions, labels, metadata and metrics; Z uses mm and A is dimensionless. |
| `training_history.npz`  | Per-epoch losses and metrics, plus scalar hyperparameters. |
| `training_config.json`  | Human-readable config and training history. |

## Stage 3 — Cluster training jobs

Submit-ready SLURM scripts for GPU partitions live in `jobs/`.

**Check available GPU partitions first:**

```bash
bash jobs/check_gpu_partitions.sh
```

It prints per-partition GPU type, idle / mix / allocated node counts, and per-node
state. Pick a partition with idle nodes and edit `--partition=` in the submit
script accordingly.

**Submit a training job:**

```bash
sbatch jobs/sub_job_elecsim_default.sh      # elecsim HDF5 input
sbatch jobs/sub_job_detsim_default.sh       # detsim HDF5 input
```

Edit `INFILE`, `OUTDIR`, and the training hyperparameters at the bottom of the
script before submission.

## Tips

- Logs go to `jobs/logs/`. Results go to `results/`. Both are ignored by git
  (see `.gitignore`).
- `radius_mm` should be tuned against the nearest-neighbor distance reported by
  `event_stats.py`. Too small -> isolated nodes; too large -> very dense graphs.
- For elecsim inputs the typical hit count is ~20 per event. For detsim with
  fine readout pitch it can be much larger and may require smaller batch sizes
  or smaller `hidden_channels`.


## Primary-electron energy asymmetry A

Each run selects one task and constructs one output head. Use `--task regression_A`
for the dimensionless 0vbb target `A = abs(E1-E2)/(E1+E2)`. The two primary electron
energies are read from the existing HDF5 primary offsets; no input conversion is
needed. Training uses `y = (A - 0.4113) / 0.2663` and the existing scalar head.

From the server project root:

```bash
mkdir -p jobs/logs
sbatch jobs/sub_job_detsim_A.sh
```

The job uses `input/processed_0vbb_above1Mev.h5`, an 80/20 split with seed 42,
and writes to `results/detsim_A_<job_id>`. It runs `results/plot_A_result.py`
after successful training. Override `INFILE`, `OUTDIR`, or `ENV_NAME` through
environment variables if needed. Cluster account/partition settings follow the
existing IHEP job and may be edited for the allocated hardware.

See [A training details](tpc_GAT/README.md#regression_a-primary-electron-energy-asymmetry)
for validation requirements, outputs, diagnostics, and checks.

## Primary-electron opening angle

Use `--task regression_theta` to reconstruct the opening-angle `cos(theta)` from
the two primary electron direction vectors. The raw label is mapped through an
empirical CDF fitted only on the training split; the saved mapping is reused for
validation and inverse transformation.

```bash
mkdir -p jobs/logs
sbatch jobs/sub_job_detsim_theta.sh
```

The job writes `results/detsim_theta_<job_id>` and generates
`theta_all_plots.pdf`, including CDF-space and raw `cos(theta)` diagnostics.

## Joint vertex XYZ reconstruction

Use `--task regression_xyz` to predict `primary_origin_x/y/z` together from
absolute hit `x y z energy log_energy` features. The existing GATv2 backbone and
xy radius graph are reused, with a three-output regression head. All target axes
share the default `845 mm` scale and `SmoothL1Loss`; predictions are saved in mm
as `[N, 3]` arrays in x, y, z order. Truth-based `--vertex_relative` and truth
vertex event features are rejected for this task.

```bash
mkdir -p jobs/logs
sbatch jobs/sub_job_detsim_vertex.sh
```

The job uses `input/detsim_tehta0vbb_noDiffusion_2mm_selected.h5`, seed 42, an
80/20 train/validation split, and an xy graph radius of 4 mm. Results are written
to `results/detsim_vertex_2mm_<job_id>`. After training, it runs
`results/plot_vertex_result.py` to produce `vertex_all_plots.pdf` and
`vertex_metrics.json`. See [joint vertex details](tpc_GAT/README.md#regression_xyz-joint-vertex-reconstruction).

## Hit-centroid frame for A / theta

Use `--vertex_centroid` to append the existing seven vertex-relative node features
with an unweighted XYZ hit centroid as reference. This supports `regression_A` and
`regression_theta`, requires `hits/z`, and does not read truth vertex fields for
input features. It is mutually exclusive with the existing `--vertex_relative`
truth-vertex option. In a truth-frame job, replace that flag and choose a separate
output directory. See [vertex frame details](tpc_GAT/README.md#vertex-relative-inputs-for-a-and-theta).
