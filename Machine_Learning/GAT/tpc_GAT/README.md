# TPC GAT Code Notes

This directory implements a task-aware graph learning pipeline based on TPC
xy-plane hit graphs. It supports the original `primary_origin_z` regression
task, primary-electron energy asymmetry A regression, opening-angle regression,
and event-id classification tasks that reuse the same graph input pattern.
Input files must follow the merged HDF5 schema defined in
`feature_extraction/HDF5_SCHEMA.md`.

## File overview

| File | Purpose |
| --- | --- |
| `tpc_graph_dataset.py` | Converts TPC HDF5 events into PyG `Data` graphs; handles hit reading, normalization, radius-graph construction, target scaling, and metadata retention. |
| `gat_model.py` | Shared edge-aware GATv2 backbone plus task-specific regression or classification heads. |
| `task_spec.py` | Task definitions, label mappings, output dimensions, and default losses. |
| `utils.py` | Training and validation helpers: DDP reduce, AMP training, regression and classification metrics. |
| `run.py` | Training entry point; supports single-process and `torchrun` DDP. |
| `event_stats.py` | Quick data inspection script for hit counts, xy nearest-neighbor distances, candidate radius edge counts, and energy / z distributions. |

## `tpc_graph_dataset.py`

`TPCGraphDataset` reads variable-length hits per event from the HDF5 file and
returns PyG graph objects.

Main fields of each sample:

| Field | Meaning |
| --- | --- |
| `x` | Node features, default `[x, y, energy, log_energy]`; `x/y` are always divided by `445 mm` and the optional `z` by `850 mm` into `[-1, 1]`, while the other features are normalized according to `feature_norm_mode`. |
| `pos` | Raw `x, y` coordinates in mm, shape `[n_hit, 2]`. |
| `edge_index` | Directed radius graph within an event. |
| `edge_attr` | `[dx/r, dy/r, distance/r, dE/std_energy, same_module]`, shape `[n_edge, 5]`. |
| `y` | Scaled training target, default `primary_origin_z / 845`, roughly in `[-1, 1]`. |
| `target_raw` | Raw Z in mm, dimensionless A, or raw classification id. |
| `event_index`, `n_hit` | Event id and filtered hit count. |
| metadata | `n_hits`, `total_energy`, `n_primaries`, `n_tracks`, `primary_origin_x/y/z`, `g4_*`, etc., kept for evaluation. |

Key configuration parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `h5_path` | required | Input HDF5 file path. |
| `node_feature_names` | `("x", "y", "energy", "log_energy")` | Hit fields used as node features; must exist in both `/hits` and `/stats/feature_names`. |
| `radius_mm` | `20.0` | Euclidean radius-graph threshold in the xy plane, in mm. |
| `target_key` | `"primary_origin_z"` | Regression target field, read from `/events/primary_origin_z` by default. |
| `target_mode` | task default | Z: `minus_one_one`, `zero_one`, `none`; A: fixed `standard`. |
| `target_abs_max` | `845.0` | Half-range of the z coordinate; under `minus_one_one`, `z_scaled = z / target_abs_max`. |
| `xy_abs_max` | `445.0` | Half-range for `x/y` geometric scaling; `x_scaled = x / xy_abs_max`, `y_scaled = y / xy_abs_max`. |
| `z_abs_max` | `850.0` | Half-range for optional `z` geometric scaling; `z_scaled = z / z_abs_max`. Only used when `z` is in `node_feature_names`. |
| `feature_norm_mode` | `"standard"` | Normalization for non-`x/y/z` node features: `standard`, `minmax`, `standard_minmax`, `none`. |
| `event_feature_keys` | `()` | Optional event-level input features from `/events/<key>`; disabled by default to avoid leaking truth information. |
| `drop_empty_events` | `True` | Whether to drop events with zero filtered hits. |

Notes:

- Graph edges are built only within a single event, never across events.
- `feature_norm_mode` does not apply to `x/y/z`; those coordinates use fixed symmetric scaling (`xy_abs_max`, `z_abs_max`). It only affects non-coordinate node features such as `energy` and `log_energy`.
- `z` is optional: add it to `--features` to feed the drift/z coordinate as a node input. It is scaled like `x/y` (mean assumed 0, symmetric range), differing only in its half-range (`850 mm`).
- `TPCData.__inc__()` prevents PyG batching from auto-offsetting `event_index`.
- `inverse_scale_target()` converts model outputs back to task units (mm for Z, dimensionless for A).
- For classification tasks, `/events/id` is read as the raw label. `binary_id`
  maps `id=1` to signal label `1` and `id=21/22/23/24` to background label `0`.
  `multiclass_id` maps `{1,21,22,23,24}` to five class indices.

## `gat_model.py`

`TPCGATModel` is a task-aware model:

1. `input_proj`: project node features to the hidden dimension.
2. Stacked `GATv2Conv(edge_dim=5, concat=False)`: use inter-hit edge features.
3. `global_mean_pool + global_max_pool`: produce an event-level graph embedding.
4. Optional `event_features` MLP: concatenate event-level input features.
5. Task head: output either one scaled regression value or classification logits.

Common parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `in_channels` | required | Node feature dimension, usually equal to `len(features)`. |
| `hidden_channels` | `128` | GAT hidden dimension. |
| `num_layers` | `3` | Number of GATv2 layers. |
| `heads` | `4` | Number of attention heads. |
| `edge_dim` | `5` | Fixed to match the dataset's edge features. |
| `event_feature_dim` | `0` | Optional event feature dimension. |
| `dropout` | `0.10` | Dropout rate. |
| `activation` | `"gelu"` | `relu`, `gelu`, `silu`. |

## `run.py`

Basic example:

```bash
torchrun --nproc_per_node=1 tpc_GAT/run.py \
  --infile /path/to/all_events.h5 \
  --outdir tpc_GAT/results \
  --task regression_z \
  --radius_mm 20 \
  --target_mode minus_one_one \
  --target_abs_max 845
```

Binary event classification example:

```bash
torchrun --nproc_per_node=1 tpc_GAT/run.py \
  --infile /path/to/all_events_with_id.h5 \
  --outdir tpc_GAT/results_binary_id \
  --task binary_id \
  --radius_mm 20
```

Main parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--infile` | required | Input HDF5 file. |
| `--outdir` | `./results_tpc_gat` | Output directory. |
| `--task` | `regression_z` | `regression_z`, `regression_A`, `regression_theta`, `binary_id`, or `multiclass_id`. |
| `--epochs` | `50` | Number of training epochs. |
| `--batch_size` | `128` | Per-process batch size. |
| `--learning_rate` | `1e-3` | OneCycleLR maximum learning rate. |
| `--weight_decay` | `1e-4` | AdamW weight decay. |
| `--features` | `x y energy log_energy` | Node feature fields. |
| `--feature_norm_mode` | `standard` | Normalization mode for non-`x/y/z` features. |
| `--radius_mm` | `20.0` | Radius-graph threshold in mm. |
| `--target_mode` | task default | Z defaults to `minus_one_one`; A uses fixed `standard`. |
| `--target_abs_max` | `845.0` | Half-range for z scaling. |
| `--xy_abs_max` | `445.0` | Half-range for x/y coordinate scaling. |
| `--z_abs_max` | `850.0` | Half-range for optional z coordinate scaling (used when `z` is in `--features`). |
| `--keep_empty_events` | off | When set, keep zero-hit events and use a dummy hit. |
| `--event_features` | empty | Optional `/events` fields fed to the model; do not add target/truth-leaking fields. |

> **Note:** Fields in `--event_features` are fed to the model as **raw values**,
> without dataset-level normalization. The `LayerNorm` in the model's internal
> `event_mlp` only normalizes within the feature dimension of a single sample,
> so it cannot remove cross-dataset scale differences (e.g. `n_hits` ranges
> roughly 0–30 while `total_energy` can reach tens of thousands). If you use
> multiple event features with very different scales, normalize them in the
> HDF5 generation stage beforehand, or use only a single feature.

| Parameter | Default | Description |
| --- | --- | --- |
| `--hidden_channels` | `128` | GAT hidden dimension. |
| `--num_layers` | `3` | Number of GATv2 layers. |
| `--heads` | `4` | Number of attention heads. |
| `--head_hidden_channels` | `128` | Regression head hidden dimension. |
| `--dropout` | `0.10` | Dropout rate. |
| `--activation` | `gelu` | Activation function. |
| `--num_workers` | `4` | Number of DataLoader workers. |
| `--grad_clip` | `1.0` | Gradient clipping norm; `<=0` disables it. |
| `--seed` | `42` | Train/validation split seed. |
| `--val_fraction` | `0.20` | Validation fraction. |
| `--log_interval` | `50` | Training log step interval. |

Output files:

| File | Contents |
| --- | --- |
| `best_model.pt` | Model weights at the epoch with the lowest validation loss. |
| `best_evaluation.npz` | Regression: `preds`, `labels`, scaled values, MAE/RMSE, and metadata. Classification: `logits`, `probabilities`, `predicted_class`, mapped `labels`, raw `id` labels, confusion matrix, metrics, and metadata. |
| `training_history.npz` | Epoch losses and task metrics, per-batch `lr`/`lr_epoch`, and configuration fields. Regression uses `val_mae`/`val_rmse`; Z also keeps `val_mae_mm`/`val_rmse_mm`. Load with `np.load("training_history.npz", allow_pickle=False)`. |
| `training_config.json` | More human-readable config and history. |

Reading the evaluation file roughly looks like:

```python
data = np.load("best_evaluation.npz")
primary_origin = data["primary_origin"]

primary_origin_x = primary_origin[:, 0]
primary_origin_y = primary_origin[:, 1]
primary_origin_z = primary_origin[:, 2]
```

## `event_stats.py`

Used for a quick pre-training check of the radius setting:

```bash
python tpc_GAT/event_stats.py /path/to/all_events.h5 --n-events 1000 --radii-mm 15,20,25
```

Output includes:

- Filtered hit count per event.
- xy nearest-neighbor distance distribution, to judge whether `radius_mm` is reasonable.
- Directed edge count at each candidate radius.
- `energy`, `log_energy`, `primary_origin_z` distributions.
- `/events/id` distribution when available, including signal/background counts
  for binary classification.
- Simple suggestions for radius and model size.

## Configurations used in the job scripts

The submit scripts in `../jobs/` use the following configurations.

**Elecsim input (`sub_job_elecsim_default.sh`)** — typically ~20 hits per event:

```bash
torchrun --nproc_per_node=1 run.py \
  --infile           all_events_elecsim_12mm.h5 \
  --outdir           results_r24_elecsim_12mm \
  --epochs           30 \
  --batch_size       16 \
  --learning_rate    1e-3 \
  --weight_decay     1e-3 \
  --features x y energy log_energy \
  --feature_norm_mode standard \
  --radius_mm        24 \
  --hidden_channels  128 \
  --num_layers       3 \
  --heads            4 \
  --num_workers      4
```

**Detsim input (`sub_job_detsim_default.sh`)** — finer readout pitch, smaller
radius and lower learning rate:

```bash
torchrun --nproc_per_node=1 run.py \
  --infile           results_detsim_12mm \
  --outdir           results_detsim_12mm \
  --epochs           30 \
  --batch_size       16 \
  --learning_rate    3e-4 \
  --weight_decay     5e-5 \
  --features x y energy log_energy \
  --feature_norm_mode standard \
  --radius_mm        12 \
  --hidden_channels  128 \
  --num_layers       3 \
  --heads            4 \
  --num_workers      4
```

If there are too few edges, try a larger `--radius_mm`; if there are too many
edges and training is slow, reduce `--radius_mm` or lower `hidden_channels`.


## regression_A: primary-electron energy asymmetry

The task name and associated scripts use an uppercase `A`. Each invocation selects
one task and one output head; this is not joint training of several targets.
The GAT backbone and scalar regression head are unchanged. The default loss remains
`SmoothL1Loss` in standardized target space.

### Labels and transforms

`events/primary_start` and `events/primary_count` locate two entries in
`primaries/primary_energy` (keV). All events must have exactly two primaries named
`e-`, valid offsets, finite nonnegative energies, and a positive energy sum.
Invalid labels raise an error with example event rows instead of silently removing
events. Targets and E1/E2 metadata are cached consistently after empty-hit selection.
No `/events/A` field is required, and `/events/id` is not used as the A target.

```text
A = abs(E1 - E2) / (E1 + E2)
y = (A - 0.4113) / 0.2663
A_pred = y_pred * 0.2663 + 0.4113
```

`--task regression_A` automatically selects `--target_mode standard`; incompatible
modes are rejected. The constants are fixed, not fitted on validation events.
Predictions are not clipped to [0, 1]; evaluation reports their out-of-range fraction.
The default input remains only `x y energy log_energy`. E1, E2 and primary/G4 truth
fields are excluded from A event inputs and retained only for labels/diagnostics.
Hit normalization still uses the input HDF5 `/stats`, as in existing tasks; this
change does not recompute hit statistics from the training split.

The inspected input contains 93842 events, no empty-hit events, and valid two-electron
labels. Its A mean/std are approximately 0.4112515/0.2662779. With the default 20%
validation fraction, the split is 75074 training and 18768 validation events.
The general loader still drops empty-hit events unless `--keep_empty_events` is set;
actual counts are recorded. Training retains the existing `drop_last=True` behavior,
so a partial training batch is omitted each epoch; validation keeps all events.

### Run and inspect

```bash
python tpc_GAT/run.py \
  --infile input/processed_0vbb_above1Mev.h5 \
  --outdir results/detsim_A_demo \
  --task regression_A --val_fraction 0.20 --seed 42 \
  --epochs 30 --batch_size 64 --learning_rate 3e-4 --weight_decay 5e-5 \
  --features x y energy log_energy --feature_norm_mode standard --radius_mm 24

python results/plot_A_result.py --indir results/detsim_A_demo
```

For cluster submission use `jobs/sub_job_detsim_A.sh` from the repository root.
Create `jobs/logs` before calling `sbatch`, because SLURM opens log files before
executing the script. The job also generates the A PDF after training.

### Output contract

- `best_model.pt`: weights from the lowest validation-loss epoch; reconstruct the
  same model using `training_config.json` for inference.
- `best_evaluation.npz`: `preds`, `labels` in A units, `preds_scaled`, `labels_scaled`,
  `mae`, `rmse`, `bias`, `residual_std`, `scaled_mae`, `out_of_range_fraction`,
  `E1`, `E2` (keV), `event_index`, and existing event metadata. Task and affine
  transform metadata are also saved. The A output does not contain `mae_mm`.
- `split_indices.npz`: original HDF5 row indices in `train_event_index` and
  `val_event_index`, plus matching raw `train_targets` and `val_targets`.
- `training_history.npz`: epoch losses and regression metrics, plus `lr` and
  `lr_epoch` for each training batch. `learning_rate` is the configured maximum,
  whereas `lr` records the rate actually used before the optimizer/scheduler step.
- `training_config.json`: all CLI options, resolved target transform, counts,
  fixed A constants, training-mean baseline and validation baseline errors,
  followed by history and best epoch after training completes.

For Z, the old `mae_mm`/`rmse_mm` evaluation and `val_mae_mm`/`val_rmse_mm` history
keys remain available as aliases. Regression validation loss is calculated over
unique events after DDP padding removal, including the partial last batch.
Python/NumPy/PyTorch seeds are initialized; exact numerical reproducibility across
GPU architectures and library versions is not guaranteed.

### A diagnostics and checks

`plot_A_result.py` checks event membership, train/validation disjointness, E1/E2 label
consistency, and inverse scaling before plotting. Its multipage PDF includes loss,
MAE versus a training-mean constant predictor, actual learning rate, and complete
validation suites for both A and `E_max = Qbb * (1 + A) / 2`. Each suite contains
split target distributions, response density, true/predicted distributions, unclipped
residuals, and binned bias/MAE/residual width versus the target, hit count, hit energy,
primary Z, and `primary R = sqrt(primary_x^2 + primary_y^2)`. `Qbb` defaults to the
median validation `E1 + E2` and can be overridden with `--Qbb_keV`.
Profiles include bin populations and retain the A=1 endpoint. Relative errors divided
by true A are deliberately avoided near A=0. Training MAE is an online training-mode
metric, not a second evaluation pass on the final epoch weights.

Run the checks in an environment with PyTorch, PyG, h5py, NumPy, Matplotlib,
scienceplots and uproot:

```bash
python -m unittest discover -s tests -v
```

The A tests create temporary HDF5 files, verify invalid-primary rejection and target
alignment, train all four task types for two tiny CPU epochs, and generate an A PDF.
They do not modify the real input or submit cluster jobs.

## regression_theta: primary-electron opening angle

`regression_theta` derives the target from the two primary direction vectors,
matching `reference/plot_0vbb_distributions.py`:

```text
cos(theta) = dot(d1, d2) / (|d1| |d2|)
y = empirical_CDF_train(cos(theta))
```

Both primaries must be electrons with finite, nonzero direction vectors. Numerical
overshoot is clipped to the physical `[-1, 1]` interval. After the seeded 80/20
split, the empirical CDF is fitted on training labels only, using mid-ranks for
ties. Its knots are stored in `split_indices.npz` and `best_evaluation.npz`; no
validation labels participate in fitting. The scalar head and `SmoothL1Loss` are
unchanged, and CDF predictions remain unclipped for CDF-space diagnostics.

```bash
python tpc_GAT/run.py --infile input/processed_0vbb_above1Mev.h5 \
  --outdir results/detsim_theta_demo --task regression_theta \
  --target_mode cdf --val_fraction 0.20 --seed 42
python results/plot_theta_result.py --indir results/detsim_theta_demo
```

The PDF contains training loss, CDF MAE, learning rate, train/validation CDF
distributions, CDF and raw `cos(theta)` response/distribution/residual plots, and
raw residual profiles versus truth, hit count, filtered energy, primary Z, and
primary radius. For SLURM use `jobs/sub_job_detsim_theta.sh`.

## regression_xyz: joint vertex reconstruction

This task reads finite `/events/primary_origin_x/y/z` truth arrays and predicts
the three coordinates jointly. Missing truth axes are errors, not zero-filled
labels. A sample has `y` and `target_raw` of shape `[1, 3]`; PyG batches and saved
`preds`, `labels`, `preds_scaled`, and `labels_scaled` have shape `[N, 3]` in x, y, z
order. The legacy scalar regression tasks retain their flat `[N]` outputs.

The graph encoder and five xy edge attributes are unchanged. Absolute node
coordinates preserve the event position: x/y are divided by 445 mm, hit-z by
850 mm. The recommended inputs are `x y z energy log_energy`. The three targets
share `target_abs_max=845 mm`, rather than separate xy/z scales, so equal physical
coordinate errors receive equal loss weights. The default target transform is
`truth / 845`, and `SmoothL1Loss` averages over events and axes. Raw labels outside
the configured scale are rejected instead of silently clipped. Outputs remain
linear and unclipped. The task also accepts the existing `zero_one` and `none`
target modes.

`--vertex_relative` is forbidden for this task because it uses the truth vertex.
Truth `primary_*` and `g4_*` event inputs are also rejected. Truth vertex metadata
is retained for diagnostics, but is not consumed by the model. `hits/z` is required
for saved geometric and energy-weighted centroid baselines. These baselines use
the same positive-energy hits as the input graph, without truth information.
In the detsim extraction code, hit-z is the energy-weighted fine-hit z within each
xy readout cell; these results describe that simulation input convention.

```bash
torchrun --standalone --nproc_per_node=1 tpc_GAT/run.py \
  --infile input/detsim_tehta0vbb_noDiffusion_2mm_selected.h5 \
  --outdir results/detsim_vertex_demo --task regression_xyz \
  --features x y z energy log_energy --radius_mm 4 \
  --target_abs_max 845 --xy_abs_max 445 --z_abs_max 850 \
  --val_fraction 0.20 --seed 42 --batch_size 64
python results/plot_vertex_result.py --indir results/detsim_vertex_demo
```

The evaluation includes per-axis `mae_xyz`, `rmse_xyz`, `bias_xyz`, and
`residual_std_xyz`, all in mm. The scalar `mae` averages the three coordinate
absolute errors; scalar `rmse` is the square root of the mean squared coordinate
error. They are distinct from `rmse_3d_mm = sqrt(mean(dx**2 + dy**2 + dz**2))`.
`mean_distance_mm`, `median_distance_mm`, `r68_mm`, and `r95_mm` summarize the
Euclidean distance distribution; R68/R95 are its 68th/95th percentiles.
Per-axis and distance metrics are also retained for each validation epoch.

The plotter verifies validation event membership, disjoint training/validation
indices, coordinate order and inverse scaling. The PDF contains loss/MAE histories,
per-axis target distributions, response densities, residuals and bias/MAE profiles,
plus the 3D error CDF and distance profiles versus transverse truth radius, truth-z,
and hit count. Profile pages include bin populations. Residuals and the CDF compare
GAT against the geometric centroid, energy-weighted centroid and training-mean
constant predictor. All numerical baseline metrics are saved in `vertex_metrics.json`.

```bash
python -m unittest discover -s tests -p 'test_regression_xyz.py' -v
```

The tests use temporary synthetic HDF5 data, check batching, gradients, physical
units, whole-event deduplication, invalid labels, truth leakage, legacy scalar
output shapes, and a two-epoch CPU training/plotting round trip. They do not train
on production events or submit jobs.

## Vertex-relative inputs for A and theta

Two mutually exclusive options append the same seven relative node features:

- `--vertex_relative`: use the MC truth vertex (existing behavior).
- `--vertex_centroid`: use the unweighted three-dimensional hit centroid, computed
  separately for each event from its retained positive-energy hits in raw mm:
  `vertex = mean(stack([hits/x, hits/y, hits/z]), axis=0)`.

The centroid is not energy-weighted and never uses `primary_origin_*`. It needs
`hits/z`, even when z is omitted from `--features`, but does not require coordinate
statistics or truth vertex fields. The new option supports `regression_A` and
`regression_theta` (both CDF and angle modes). With neither option, no relative
features are appended. The centroid is a reference point, not a claim of the true
interaction vertex.

Both modes append `[v_dx, v_dy, v_dz, v_r, v_ux, v_uy, v_uz]`. Displacements use
`xy_abs_max`/`z_abs_max`; directions use raw mm displacement. Existing absolute
node features and the xy radius graph are retained, matching the truth-frame
implementation. Default empty-event filtering is unchanged; explicitly kept empty
events use the existing zero-valued dummy hit, giving a zero centroid.

For example, add `--vertex_centroid` to an A or theta training command:

```bash
python tpc_GAT/run.py --infile input/detsim_tehta0vbb_noDiffusion_2mm_selected.h5 \
  --outdir results/detsim_A_centroid --task regression_A \
  --features x y z energy log_energy --radius_mm 4 --vertex_centroid
python tpc_GAT/run.py --infile input/detsim_tehta0vbb_noDiffusion_2mm_selected.h5 \
  --outdir results/detsim_theta_centroid --task regression_theta --target_mode angle \
  --features x y z energy log_energy --radius_mm 4 --vertex_centroid
```

In existing truth-frame jobs, replace `--vertex_relative` with
`--vertex_centroid`; use a separate output directory when comparing modes.
The training configuration/history record `vertex_centroid` and the resolved
`vertex_source` (`none`, `truth`, or `centroid`). Evaluation saves the actual
reference points as `frame_vertex` with shape `[N, 3]` in mm for either frame mode.
Old checkpoints can retain the same feature width, but centroid-frame models
should be trained and evaluated consistently with the new input convention.
