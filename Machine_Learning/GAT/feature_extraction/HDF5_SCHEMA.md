# HDF5 schema for ML inputs

`root_to_hdf5.py` converts each processed ROOT `HitTree` entry into one event in HDF5.
Hits with `energy <= 0` are removed before writing hit arrays and before computing normalization statistics.

## h5 file structure

file.h5
├── attrs
│   ├── schema_version = "1.0"
│   ├── source_root = "..."
│   ├── tree_name = "HitTree"
│   ├── filter = "energy > 0"
│   ├── hit_feature_names = ["x", "y", "z", "energy", "log_energy"]
│   └── graph_coordinate_names = ["x", "y", "local_q", "local_r", "module_id"]
│
├── events/
│   ├── event_id
│   ├── hit_start
│   ├── hit_count
│   ├── primary_start
│   ├── primary_count
│   ├── n_hits
│   ├── n_hits_original
│   ├── id                         (optional classification label)
│   ├── total_energy
│   ├── total_energy_filtered
│   ├── n_primaries
│   ├── n_tracks
│   ├── primary_origin_x/y/z
│   ├── g4_total_deposited_energy
│   └── g4_sensitive_volume_energy
│
├── hits/
│   ├── event_id
│   ├── x, y, z
│   ├── energy
│   ├── log_energy
│   ├── module_id, module_col, module_row
│   ├── module_q, module_r
│   ├── local_col, local_row
│   ├── local_q, local_r
│   ├── q, r
│   ├── valid_geometry
│   ├── pad_center_x, pad_center_y
│   ├── residual_x, residual_y
│   └── nearest_pad_distance
│
├── primaries/
│   ├── event_id
│   ├── primary_particle_name
│   ├── primary_energy
│   └── primary_dir_x/y/z
│
├── stats/
│   ├── feature_names
│   ├── mean
│   ├── std
│   ├── min
│   └── max
│
└── summary/
    └── attrs
        ├── n_events
        ├── n_hits_before_filter
        ├── n_hits_after_filter
        └── n_primaries


## Groups

- `/events`: event-level metadata and offsets into hit/primary arrays.
  - event index = row index (no `event_id` stored; use row position as the unique key).
  - `hit_start`, `hit_count`: slice into `/hits/*` for each event.
  - `primary_start`, `primary_count`: slice into `/primaries/*` for each event.
  - `id`: optional event-level classification label. Current convention:
    `1 = 2nubb_ex` signal, `21 = 2nubb_gs`, `22 = co60`, `23 = u238`,
    `24 = th232` backgrounds.
  - `n_hits_original`: hit count before `energy > 0` filtering.
  - `total_energy`: original value from the ROOT file.
  - `total_energy_filtered`: sum of positive-energy hits after filtering.
  - truth fields: `n_primaries`, `n_tracks`, `primary_origin_x/y/z`, `g4_total_deposited_energy`, `g4_sensitive_volume_energy`.
- `/hits`: flat positive-energy hit table.
  - point-cloud features: `x`, `y`, optional drift-coordinate feature `z`, `energy`, `log_energy`.
  - graph coordinate alternatives: `x`, `y`, `local_q`, `local_r`, `module_id`.
  - extra geometry fields are preserved: `module_col`, `module_row`, `module_q`, `module_r`, `local_col`, `local_row`, `q`, `r`, `valid_geometry`, pad center and residual fields.
- `/primaries`: flat primary-particle table.
  - `primary_particle_name`, `primary_energy`, `primary_dir_x/y/z`.
- `/stats`: normalization statistics over all positive-energy hits.
  - `feature_names`: `x`, `y`, `z`, `local_q`, `local_r`, `module_id`, `energy`, `log_energy`.
  - `mean`, `std`, `min`, `max`; `std` is population standard deviation.
- `/summary`: file-level counts.

## Typical use

For event `i`, load hits with:

```python
start = h5["events/hit_start"][i]
count = h5["events/hit_count"][i]
points = h5["hits/x"][start:start + count], h5["hits/y"][start:start + count]
energy = h5["hits/energy"][start:start + count]

p_start = h5["events/primary_start"][i]
p_count = h5["events/primary_count"][i]
primary_name = h5["primaries/primary_particle_name"][p_start:p_start + p_count]
primary_dir_x = h5["primaries/primary_dir_x"][p_start:p_start + p_count]
```

For a GAT model, build `radius_graph` from `/hits/x` and `/hits/y` inside one event.
When `z` is selected as a node feature, normalize it through `/stats` like
`energy` and `log_energy`; the training dataset can also add `dz/std(z)` as an
edge feature. For a point-cloud model, stack `/hits/x`, `/hits/y`, optionally
`/hits/z`, `/hits/energy`, and `/hits/log_energy`.
