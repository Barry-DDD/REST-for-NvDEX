# GAT for NvDEx

Reconstruction of the primary vertex Z position (`primary_origin_z`) of NvDEx TPC
events using a Graph Attention Network (GATv2) on the xy-plane hits.

Can be extended to other tasks.

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
        |  tpc_GAT/run.py        (GATv2 training)
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
│   ├── gat_model.py                      Edge-aware GATv2 regressor
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

The project runs on SCNC HPC cluster (CentOS 7).

| Stage | Environment |
| --- | --- |
| ROOT extraction (`extract_*.C`) | REST + ROOT, loaded via `source ~/env_rest.sh` |
| HDF5 conversion / merging       | Python + `h5py` + `uproot`, conda env `pyg-env` |
| Training (`tpc_GAT/run.py`)     | Python + PyTorch + PyTorch Geometric, conda env `pyg-env` |

Activate conda with:

```bash
source ~/env_conda.sh
conda activate pyg-env
```

## Stage 1 — Feature extraction (REST ROOT -> HDF5)

### 1.1 REST ROOT -> processed TTree

Two macros are provided, depending on which simulation stage you want as input.

**Full electronics simulation:**

```bash
source ~/env_rest.sh
restRoot -b -q 'feature_extraction/extract_elecsim_hits.C("input.root","processed.root")'
```

**Detector-level fine hits rebinned to a chosen readout pitch (mm):**

```bash
source ~/env_rest.sh
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

The model implements an edge-aware GATv2 regression network that takes per-event
hit graphs and predicts the scaled `primary_origin_z`. See
[`tpc_GAT/README.md`](tpc_GAT/README.md) for the complete model documentation,
dataset fields, hyperparameters, and output file formats.

### Quick start

```bash
conda activate pyg-env

torchrun --nproc_per_node=1 tpc_GAT/run.py \
  --infile  /path/to/all_events.h5 \
  --outdir  tpc_GAT/results/run_demo \
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
| `best_evaluation.npz`   | Per-event predictions (mm), labels (mm), metadata, plus MAE / RMSE. |
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
