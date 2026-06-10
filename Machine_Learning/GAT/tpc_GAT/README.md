# TPC GAT Code Notes

本目录实现基于 TPC xy 平面 hit 图的 `primary_origin_z` 回归训练流程。输入文件应符合 `tpc_feature_extraction/HDF5_SCHEMA.md` 中定义的合并 HDF5 schema。

## 文件概览

| 文件 | 作用 |
| --- | --- |
| `tpc_graph_dataset.py` | 将 TPC HDF5 event 转换为 PyG `Data` 图，负责 hit 读取、归一化、半径图构建、target 缩放和 metadata 保留。 |
| `gat_model.py` | 轻量 Edge-aware GATv2 回归模型，核心层为 `GATv2Conv(edge_dim=5)`。 |
| `utils.py` | 训练和验证辅助函数，包含 DDP reduce、AMP 训练、回归指标统计、scaled/raw target 转换。 |
| `run.py` | 训练入口，持单进程和 `torchrun` DDP。 |
| `event_stats.py` | 数据快速统计脚本，用于查看 hit 数、xy 最近邻距离、候选半径边数、能量和 z 分布。 |

## `tpc_graph_dataset.py`

`TPCGraphDataset` 从 HDF5 中按 event 读取变长 hit，并返回 PyG 图对象。

每个样本主要字段：

| 字段 | 含义 |
| --- | --- |
| `x` | node feature，默认 `[x, y, energy, log_energy]`；其中 `x/y` 固定除以 `445 mm` 到 `[-1, 1]`，其他 feature 按 `feature_norm_mode` 归一化。 |
| `pos` | raw `x, y` 坐标，单位 mm，shape `[n_hit, 2]`。 |
| `edge_index` | event 内 directed radius graph。 |
| `edge_attr` | `[dx/r, dy/r, distance/r, dE/std_energy, same_module]`，shape `[n_edge, 5]`。 |
| `y` | 缩放后的训练 target，默认 `primary_origin_z / 845`，范围约 `[-1, 1]`。 |
| `target_raw` | 原始 `primary_origin_z`，单位 mm。 |
| `event_index`, `n_hit` | event id 和 filtered hit 数。 |
| metadata | `n_hits`, `total_energy`, `n_primaries`, `n_tracks`, `primary_origin_x/y/z`, `g4_*` 等评估用信息。 |

关键设置参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `h5_path` | 必填 | 输入 HDF5 文件路径。 |
| `node_feature_names` | `("x", "y", "energy", "log_energy")` | 作为 node feature 的 hit 字段，必须同时存在于 `/hits` 和 `/stats/feature_names`。 |
| `radius_mm` | `20.0` | xy 平面欧氏半径图阈值，单位 mm。 |
| `target_key` | `"primary_origin_z"` | 回归目标字段，默认从 `/events/primary_origin_z` 读取。 |
| `target_mode` | `"minus_one_one"` | target 缩放方式：`minus_one_one`, `zero_one`, `none`。 |
| `target_abs_max` | `845.0` | z 坐标半范围；`minus_one_one` 下 `z_scaled = z / target_abs_max`。 |
| `xy_abs_max` | `445.0` | `x/y` 几何坐标缩放半范围；`x_scaled = x / xy_abs_max`, `y_scaled = y / xy_abs_max`。 |
| `feature_norm_mode` | `"standard"` | 非 `x/y` node feature 归一化：`standard`, `minmax`, `standard_minmax`, `none`。 |
| `event_feature_keys` | `()` | 可选 event-level 输入特征，来自 `/events/<key>`；默认关闭，避免误用 truth 信息。 |
| `drop_empty_events` | `True` | 是否丢弃 filtered hit 数为 0 的 event。 |

注意：

- 图边只在单个 event 内构造，不跨 event。
- `feature_norm_mode` 不作用于 `x/y`；只作用于 `energy`, `log_energy` 等非坐标 node feature。
- `TPCData.__inc__()` 避免 `event_index` 被 PyG batching 自动偏移。
- `inverse_scale_target()` 可将模型输出从 scaled value 转回 mm。



## `gat_model.py`

`TPCGATRegressor` 是轻量回归模型：

1. `input_proj`: node feature 投影到 hidden dimension。
2. 多层 `GATv2Conv(edge_dim=5, concat=False)`：使用 hit 间边特征。
3. `global_mean_pool + global_max_pool`：得到 event-level graph embedding。
4. 可选 `event_features` MLP：拼接事例级输入特征。
5. regression head：输出一个 scaled `primary_origin_z`。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `in_channels` | 必填 | node feature 维度，通常等于 `len(features)`。 |
| `hidden_channels` | `128` | GAT hidden dimension。 |
| `num_layers` | `3` | GATv2 层数。 |
| `heads` | `4` | attention heads。 |
| `edge_dim` | `5` | 固定匹配 dataset 的 edge feature。 |
| `event_feature_dim` | `0` | 可选 event feature 维度。 |
| `dropout` | `0.10` | dropout rate。 |
| `activation` | `"gelu"` | `relu`, `gelu`, `silu`。 |

## `run.py`

基础示例：

```bash
torchrun --nproc_per_node=1 tpc_GAT/run.py \
  --infile /path/to/all_events.h5 \
  --outdir tpc_GAT/results \
  --radius_mm 20 \
  --target_mode minus_one_one \
  --target_abs_max 845
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--infile` | 必填 | 输入 HDF5 文件。 |
| `--outdir` | `./results_tpc_gat` | 输出目录。 |
| `--epochs` | `50` | 训练 epoch 数。 |
| `--batch_size` | `128` | 每个进程 batch size。 |
| `--learning_rate` | `1e-3` | OneCycleLR 最大学习率。 |
| `--weight_decay` | `1e-4` | AdamW weight decay。 |
| `--features` | `x y energy log_energy` | node feature 字段。 |
| `--feature_norm_mode` | `standard` | 非 `x/y` feature 归一化方式。 |
| `--radius_mm` | `20.0` | radius graph 阈值，单位 mm。 |
| `--target_mode` | `minus_one_one` | target 缩放方式。 |
| `--target_abs_max` | `845.0` | z 缩放半范围。 |
| `--xy_abs_max` | `445.0` | x/y 坐标缩放半范围。 |
| `--keep_empty_events` | off | 打开后保留 zero-hit event，并使用 dummy hit。 |
| `--event_features` | 空 | 可选 `/events` 字段输入模型；不要加入 target/truth 泄漏字段。 |

> **注意：** `--event_features` 中的字段以**原始值**送入模型，不做 dataset 级别的归一化。模型内部 `event_mlp` 的 `LayerNorm` 只在单个样本的特征维度内归一化，无法消除跨 dataset 的量纲差异（例如 `n_hits` 范围约 0–30，`total_energy` 范围可达数万）。如果同时使用多个量纲差异较大的 event feature，建议在 HDF5 生成阶段预先归一化，或只使用单个 feature。
| `--hidden_channels` | `128` | GAT hidden dimension。 |
| `--num_layers` | `3` | GATv2 层数。 |
| `--heads` | `4` | attention heads。 |
| `--head_hidden_channels` | `128` | regression head hidden dimension。 |
| `--dropout` | `0.10` | dropout rate。 |
| `--activation` | `gelu` | activation。 |
| `--num_workers` | `4` | DataLoader worker 数。 |
| `--grad_clip` | `1.0` | gradient clipping norm；`<=0` 关闭。 |
| `--seed` | `42` | train/validation split seed。 |
| `--val_fraction` | `0.20` | validation fraction。 |
| `--log_interval` | `50` | 训练日志 step 间隔。 |

输出文件：

| 文件 | 内容 |
| --- | --- |
| `best_model.pt` | validation loss 最低 epoch 的模型权重。 |
| `best_evaluation.npz` | `preds`, `labels` 为反缩放后的 mm 值；同时保存 `preds_scaled`, `labels_scaled`, `mae_mm`, `rmse_mm` 和 metadata。 |
| `training_history.npz` | Structured NumPy archive: per-epoch `train_loss`, `val_loss`, `train_scaled_mae`, `val_scaled_mae`, `val_mae_mm`, `val_rmse_mm`, and scalar hyperparameters. Load with `np.load("training_history.npz")`. String fields (feature names, norm mode, etc.) are in `training_config.json`. |
| `training_config.json` | 可读性更好的配置和历史记录。 |


读取时大概这样：
```
  data = np.load("best_evaluation.npz")
  primary_origin = data["primary_origin"]

  primary_origin_x = primary_origin[:, 0]
  primary_origin_y = primary_origin[:, 1]
  primary_origin_z = primary_origin[:, 2]
```

## `event_stats.py`

用于训练前快速检查半径设置：

```bash
python tpc_GAT/event_stats.py /path/to/all_events.h5 --n-events 1000 --radii-mm 15,20,25
```

输出包括：

- 每个 event 的 filtered hit count。
- xy 最近邻距离分布，用于判断 `radius_mm` 是否合理。
- 每个候选 radius 下的 directed edge count。
- `energy`, `log_energy`, `primary_origin_z` 分布。
- 简单的半径和模型规模建议。

## 建议默认起点

如果每个 event 约 20 个 hit，先从以下配置开始：

```bash
torchrun --nproc_per_node=1 run.py \
  --infile /public/home/liuz1/work/26.03.18_rest/test_example/feature_extraction/all_events.h5 \
  --outdir results_r24 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --features x y energy log_energy \
  --feature_norm_mode standard \
  --xy_abs_max 445 \
  --target_abs_max 845.0 \
  --radius_mm 24 \
  --hidden_channels 128 \
  --num_layers 3 \
  --heads 4 
```

如果边数过少，尝试 `--radius_mm 25`；如果边数过多且训练慢，尝试 `--radius_mm 15` 或降低 `hidden_channels`。
