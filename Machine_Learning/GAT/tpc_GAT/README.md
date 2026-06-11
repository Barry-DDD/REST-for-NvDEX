# TPC GAT Code Notes

This directory implements the `primary_origin_z` regression training pipeline
based on graphs of TPC xy-plane hits. Input files must follow the merged HDF5
schema defined in `tpc_feature_extraction/HDF5_SCHEMA.md`.

## File overview

| File | Purpose |
| --- | --- |
| `tpc_graph_dataset.py` | Converts TPC HDF5 events into PyG `Data` graphs; handles hit reading, normalization, radius-graph construction, target scaling, and metadata retention. |
| `gat_model.py` | Lightweight edge-aware GATv2 regression model; the core layer is `GATv2Conv(edge_dim=5)`. |
| `utils.py` | Training and validation helpers: DDP reduce, AMP training, regression metrics, and scaled/raw target conversion. |
| `run.py` | Training entry point; supports single-process and `torchrun` DDP. |
| `event_stats.py` | Quick data inspection script for hit counts, xy nearest-neighbor distances, candidate radius edge counts, and energy / z distributions. |

## `tpc_graph_dataset.py`

`TPCGraphDataset` reads variable-length hits per event from the HDF5 file and
returns PyG graph objects.

Main fields of each sample:

| Field | Meaning |
| --- | --- |
| `x` | Node features, default `[x, y, energy, log_energy]`; `x/y` are always divided by `445 mm` into `[-1, 1]`, while the other features are normalized according to `feature_norm_mode`. |
| `pos` | Raw `x, y` coordinates in mm, shape `[n_hit, 2]`. |
| `edge_index` | Directed radius graph within an event. |
| `edge_attr` | `[dx/r, dy/r, distance/r, dE/std_energy, same_module]`, shape `[n_edge, 5]`. |
| `y` | Scaled training target, default `primary_origin_z / 845`, roughly in `[-1, 1]`. |
| `target_raw` | Raw `primary_origin_z` in mm. |
| `event_index`, `n_hit` | Event id and filtered hit count. |
| metadata | `n_hits`, `total_energy`, `n_primaries`, `n_tracks`, `primary_origin_x/y/z`, `g4_*`, etc., kept for evaluation. |

Key configuration parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `h5_path` | required | Input HDF5 file path. |
| `node_feature_names` | `("x", "y", "energy", "log_energy")` | Hit fields used as node features; must exist in both `/hits` and `/stats/feature_names`. |
| `radius_mm` | `20.0` | Euclidean radius-graph threshold in the xy plane, in mm. |
| `target_key` | `"primary_origin_z"` | Regression target field, read from `/events/primary_origin_z` by default. |
| `target_mode` | `"minus_one_one"` | Target scaling: `minus_one_one`, `zero_one`, `none`. |
| `target_abs_max` | `845.0` | Half-range of the z coordinate; under `minus_one_one`, `z_scaled = z / target_abs_max`. |
| `xy_abs_max` | `445.0` | Half-range for `x/y` geometric scaling; `x_scaled = x / xy_abs_max`, `y_scaled = y / xy_abs_max`. |
| `feature_norm_mode` | `"standard"` | Normalization for non-`x/y` node features: `standard`, `minmax`, `standard_minmax`, `none`. |
| `event_feature_keys` | `()` | Optional event-level input features from `/events/<key>`; disabled by default to avoid leaking truth information. |
| `drop_empty_events` | `True` | Whether to drop events with zero filtered hits. |

Notes:

- Graph edges are built only within a single event, never across events.
- `feature_norm_mode` does not apply to `x/y`; it only affects non-coordinate node features such as `energy` and `log_energy`.
- `TPCData.__inc__()` prevents PyG batching from auto-offsetting `event_index`.
- `inverse_scale_target()` converts model outputs from scaled values back to mm.

## `gat_model.py`

`TPCGATRegressor` is a lightweight regression model:

1. `input_proj`: project node features to the hidden dimension.
2. Stacked `GATv2Conv(edge_dim=5, concat=False)`: use inter-hit edge features.
3. `global_mean_pool + global_max_pool`: produce an event-level graph embedding.
4. Optional `event_features` MLP: concatenate event-level input features.
5. Regression head: output a single scaled `primary_origin_z`.

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
  --radius_mm 20 \
  --target_mode minus_one_one \
  --target_abs_max 845
```

Main parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--infile` | required | Input HDF5 file. |
| `--outdir` | `./results_tpc_gat` | Output directory. |
| `--epochs` | `50` | Number of training epochs. |
| `--batch_size` | `128` | Per-process batch size. |
| `--learning_rate` | `1e-3` | OneCycleLR maximum learning rate. |
| `--weight_decay` | `1e-4` | AdamW weight decay. |
| `--features` | `x y energy log_energy` | Node feature fields. |
| `--feature_norm_mode` | `standard` | Normalization mode for non-`x/y` features. |
| `--radius_mm` | `20.0` | Radius-graph threshold in mm. |
| `--target_mode` | `minus_one_one` | Target scaling mode. |
| `--target_abs_max` | `845.0` | Half-range for z scaling. |
| `--xy_abs_max` | `445.0` | Half-range for x/y coordinate scaling. |
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
| `best_evaluation.npz` | `preds`, `labels` are inverse-scaled mm values; also stores `preds_scaled`, `labels_scaled`, `mae_mm`, `rmse_mm`, and metadata. |
| `training_history.npz` | Structured NumPy archive: per-epoch `train_loss`, `val_loss`, `train_scaled_mae`, `val_scaled_mae`, `val_mae_mm`, `val_rmse_mm`, and scalar hyperparameters. Load with `np.load("training_history.npz")`. String fields (feature names, norm mode, etc.) are in `training_config.json`. |
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
