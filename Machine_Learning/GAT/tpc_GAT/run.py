"""Training entry point for task-aware TPC GAT training."""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from gat_model import TPCGATModel
from task_spec import (SUPPORTED_TASKS, VERTEX_METRIC_KEYS, VERTEX_TARGET_KEYS,
                       fit_cdf_transform, get_task_spec, get_target_transform)
from tpc_graph_dataset import TPCGraphDataset
from utils import train_epoch, validate_epoch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TPC Edge-aware GATv2 training for regression or event classification")
    parser.add_argument("--infile", type=str, required=True, help="Input merged TPC HDF5 file path.")
    parser.add_argument("--outdir", default="./results_tpc_gat", type=str, help="Output directory.")
    parser.add_argument("--task", default="regression_z", choices=SUPPORTED_TASKS, help="Training task.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per process.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Maximum learning rate for OneCycleLR.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay for AdamW.")
    parser.add_argument("--features", nargs="+", default=["x", "y", "energy", "log_energy"], help="Selected hit node features.")
    parser.add_argument("--feature_norm_mode", default="standard", choices=["standard", "minmax", "standard_minmax", "none"])
    parser.add_argument("--radius_mm", default=20.0, type=float, help="Radius graph threshold in raw xy mm.")
    parser.add_argument("--target_mode", default=None, choices=["minus_one_one", "zero_one", "none", "standard", "cdf", "angle"], help="Default: minus_one_one for Z/XYZ; standard for A; cdf (cos theta) for theta. Use angle to regress theta in degrees linearly scaled to [-1, 1].")
    parser.add_argument("--target_abs_max", default=845.0, type=float, help="Vertex target scale in mm; XYZ uses the same scale for all three axes.")
    parser.add_argument("--xy_abs_max", default=445.0, type=float, help="Absolute xy coordinate range used for x/y scaling.")
    parser.add_argument("--z_abs_max", default=850.0, type=float, help="Absolute z coordinate range used for z scaling.")
    parser.add_argument("--keep_empty_events", action="store_true", help="Keep zero-hit events as one dummy zero hit.")
    parser.add_argument("--event_features", nargs="*", default=[], help="Optional /events feature keys concatenated before the task head.")
    vertex_group = parser.add_mutually_exclusive_group()
    vertex_group.add_argument("--vertex_relative", action="store_true", help="Append relative features using the truth vertex.")
    vertex_group.add_argument("--vertex_centroid", action="store_true", help="Append relative features using the unweighted XYZ hit centroid; for regression_A / regression_theta.")
    parser.add_argument("--hidden_channels", default=128, type=int, help="GAT hidden channel dimension.")
    parser.add_argument("--num_layers", default=3, type=int, help="Number of GATv2Conv layers.")
    parser.add_argument("--heads", default=4, type=int, help="Number of attention heads.")
    parser.add_argument("--head_hidden_channels", default=128, type=int, help="Task head hidden dimension.")
    parser.add_argument("--dropout", default=0.10, type=float, help="Dropout probability.")
    parser.add_argument("--activation", default="gelu", choices=["relu", "gelu", "silu"])
    parser.add_argument("--num_workers", default=4, type=int, help="Number of DataLoader workers per process.")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="Gradient clipping norm. Set <= 0 to disable.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for train/validation split.")
    parser.add_argument("--val_fraction", default=0.20, type=float, help="Validation split fraction.")
    parser.add_argument("--log_interval", default=50, type=int, help="Training log interval.")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training when launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    return distributed, rank, world_size, local_rank, device


def print_config(args, world_size, input_channels, event_feature_dim, task_spec, target_transform):
    """Print the training configuration."""
    print("========== TPC GAT Training Configuration ==========")
    print(f"Input file: {args.infile}")
    print(f"Output directory: {args.outdir}")
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size per process: {args.batch_size}")
    print(f"World size: {world_size}")
    print(f"Effective global batch size: {args.batch_size * world_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Node features: {args.features}")
    print(f"Vertex frame: {'centroid' if args.vertex_centroid else 'truth' if args.vertex_relative else 'none'}")
    print(f"Input channels: {input_channels}")
    print(f"Feature normalization mode: {args.feature_norm_mode}")
    print(f"XY coordinate scale: +/- {args.xy_abs_max} mm -> [-1, 1]")
    print(f"Z coordinate scale: +/- {args.z_abs_max} mm -> [-1, 1]")
    print(f"Radius graph: {args.radius_mm} mm")
    if task_spec.is_classification:
        print(f"Target: {task_spec.target_key}, classes: {task_spec.num_outputs}")
    else:
        print(f"Target: {task_spec.target_key}, mode={target_transform.mode}, unit={target_transform.unit}")
    print(f"Event feature keys: {args.event_features}")
    print(f"Event feature dimension: {event_feature_dim}")
    print(f"GAT hidden/layers/heads: {args.hidden_channels}/{args.num_layers}/{args.heads}")
    print("====================================================")


def save_best_result(outdir, model, val_result, task_spec, target_transform=None):
    """Save the best model and corresponding validation result."""
    model_path = os.path.join(outdir, "best_model.pt")
    eval_path = os.path.join(outdir, "best_evaluation.npz")
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, model_path)

    if task_spec.is_classification:
        save_dict = {
            "logits": val_result["logits"],
            "probabilities": val_result["probabilities"],
            "predicted_class": val_result["predicted_class"],
            "labels": val_result["labels"],
            "raw_labels": val_result["raw_labels"],
            "confusion_matrix": val_result["confusion_matrix"],
            "accuracy": np.asarray([val_result["accuracy"]], dtype=np.float32),
            "precision": np.asarray([val_result["precision"]], dtype=np.float32),
            "recall": np.asarray([val_result["recall"]], dtype=np.float32),
            "f1": np.asarray([val_result["f1"]], dtype=np.float32),
        }
        if "roc_auc" in val_result:
            save_dict["roc_auc"] = np.asarray([val_result["roc_auc"]], dtype=np.float32)
    else:
        save_dict = {
            "preds": val_result["preds"],
            "labels": val_result["labels"],
            "preds_scaled": val_result["preds_scaled"],
            "labels_scaled": val_result["labels_scaled"],
            "mae": np.asarray([val_result["mae"]], dtype=np.float32),
            "rmse": np.asarray([val_result["rmse"]], dtype=np.float32),
            "bias": np.asarray([val_result["bias"]], dtype=np.float32),
            "residual_std": np.asarray([val_result["residual_std"]], dtype=np.float32),
            "scaled_mae": np.asarray([val_result["scaled_mae"]], dtype=np.float32),
        }
    for key in ("mae_mm", "rmse_mm", "out_of_range_fraction", "cdf_out_of_range_fraction"):
        if key in val_result:
            save_dict[key] = np.asarray([val_result[key]], dtype=np.float32)
    save_dict["task"] = np.asarray(task_spec.name)
    if task_spec.name == "regression_xyz":
        save_dict["target_names"] = np.asarray(VERTEX_TARGET_KEYS)
        for key in VERTEX_METRIC_KEYS:
            save_dict[key] = np.asarray(val_result[key], dtype=np.float64)
    if target_transform is not None:
        save_dict.update(target_mode=np.asarray(target_transform.mode),
                         target_offset=np.asarray(target_transform.offset),
                         target_scale=np.asarray(target_transform.scale),
                         target_unit=np.asarray(target_transform.unit))
        if target_transform.mode == "cdf":
            save_dict.update(cdf_values=target_transform.cdf_values,
                             cdf_probabilities=target_transform.cdf_probabilities)
    for key, value in val_result["metadata"].items():
        save_dict[key] = value
    np.savez_compressed(eval_path, **save_dict)


def stratified_split_indices(labels, val_fraction, seed):
    """Return train/validation indices preserving class fractions as closely as possible."""
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(int(seed))
    train_indices = []
    val_indices = []
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)[0]
        rng.shuffle(indices)
        n_val = max(1, int(round(indices.size * float(val_fraction)))) if indices.size > 1 else 1
        n_val = min(n_val, indices.size - 1) if indices.size > 1 else 1
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    if not train_indices:
        raise ValueError("Stratified split produced zero training samples. Increase dataset size or reduce val_fraction.")
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def main():
    """Run training."""
    args = parse_args()
    task_spec = get_task_spec(args.task)
    target_transform = get_target_transform(args.task, args.target_mode, args.target_abs_max)
    args.target_mode = target_transform.mode
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")

    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(args.outdir, exist_ok=True)
    if distributed:
        torch.distributed.barrier()

    full_data = TPCGraphDataset(
        h5_path=args.infile,
        node_feature_names=args.features,
        radius_mm=args.radius_mm,
        task=args.task,
        target_mode=args.target_mode,
        target_abs_max=args.target_abs_max,
        xy_abs_max=args.xy_abs_max,
        z_abs_max=args.z_abs_max,
        feature_norm_mode=args.feature_norm_mode,
        event_feature_keys=args.event_features,
        vertex_relative=args.vertex_relative,
        vertex_centroid=args.vertex_centroid,
        drop_empty_events=(not args.keep_empty_events),
    )

    input_channels = full_data.node_feature_dim
    event_feature_dim = full_data.event_feature_dim
    if task_spec.is_classification:
        train_indices, val_indices = stratified_split_indices(full_data.labels, args.val_fraction, args.seed)
        train_dataset = Subset(full_data, train_indices)
        val_dataset = Subset(full_data, val_indices)
    else:
        val_size = max(1, int(round(len(full_data) * args.val_fraction)))
        train_size = len(full_data) - val_size
        if train_size <= 0:
            raise ValueError(f"Invalid split sizes: train_size={train_size}, val_size={val_size}.")
        split_generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_data, [train_size, val_size], generator=split_generator)

    if args.task == "regression_theta" and target_transform.mode == "cdf":
        target_transform = fit_cdf_transform(full_data.raw_targets[np.asarray(train_dataset.indices)])
        full_data.set_target_transform(target_transform)

    if is_main_process:
        print_config(args, world_size, input_channels, event_feature_dim, task_spec, target_transform)

    split_info = {
        "train_event_index": full_data.event_indices[np.asarray(train_dataset.indices)],
        "val_event_index": full_data.event_indices[np.asarray(val_dataset.indices)],
        "train_targets": full_data.raw_targets[np.asarray(train_dataset.indices)],
        "val_targets": full_data.raw_targets[np.asarray(val_dataset.indices)],
    }
    if is_main_process:
        np.savez_compressed(os.path.join(args.outdir, "split_indices.npz"), **split_info)
    run_config = dict(vars(args))
    run_config["vertex_source"] = full_data.vertex_source
    run_config.update(target_key=full_data.target_key, target_unit=target_transform.unit,
                      target_offset=target_transform.offset, target_scale=target_transform.scale,
                      total_events=full_data.total_events, usable_events=len(full_data),
                      train_size=len(train_dataset), val_size=len(val_dataset), world_size=world_size)
    if args.task == "regression_xyz":
        run_config.update(target_names=list(VERTEX_TARGET_KEYS),
                          metric_definition="mae/rmse average coordinate errors; rmse_3d_mm uses Euclidean distance")
    if args.task == "regression_A":
        baseline = float(np.mean(split_info["train_targets"], dtype=np.float64))
        residual = baseline - split_info["val_targets"].astype(np.float64)
        run_config.update(A_mu=target_transform.offset, A_sigma=target_transform.scale,
                          target_formula="abs(E1-E2)/(E1+E2)", baseline_A=baseline,
                          baseline_mae=float(np.mean(np.abs(residual))),
                          baseline_rmse=float(np.sqrt(np.mean(residual ** 2))))
    if args.task == "regression_theta" and target_transform.mode == "angle":
        baseline = float(np.mean(split_info["train_targets"], dtype=np.float64))
        residual = baseline - split_info["val_targets"].astype(np.float64)
        run_config.update(target_formula="degrees(arccos(dot(d1,d2)/(|d1||d2|)))", baseline_theta_deg=baseline,
                          baseline_mae=float(np.mean(np.abs(residual))),
                          baseline_rmse=float(np.sqrt(np.mean(residual ** 2))))
    if args.task == "regression_theta" and target_transform.mode == "cdf":
        baseline = float(np.mean(split_info["train_targets"], dtype=np.float64))
        residual = baseline - split_info["val_targets"].astype(np.float64)
        run_config.update(target_formula="dot(d1,d2)/(|d1||d2|)", cdf_fit="training_split_only",
                          cdf_knot_count=int(target_transform.cdf_values.size), baseline_cos_theta=baseline,
                          baseline_mae=float(np.mean(np.abs(residual))),
                          baseline_rmse=float(np.sqrt(np.mean(residual ** 2))))
        split_info.update(cdf_values=target_transform.cdf_values,
                          cdf_probabilities=target_transform.cdf_probabilities)
        if is_main_process:
            np.savez_compressed(os.path.join(args.outdir, "split_indices.npz"), **split_info)
    if is_main_process:
        with open(os.path.join(args.outdir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(run_config, handle, indent=2)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    if len(train_loader) <= 0:
        raise ValueError("The training DataLoader has zero batches. Reduce batch size/world size or increase dataset size.")

    if is_main_process:
        sample = full_data[0]
        print(f"Dataset size after hit selection: {len(full_data)}")
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of val_loader: {len(val_loader)}")
        print(f"Sample x shape: {tuple(sample.x.shape)}")
        print(f"Sample edge_index shape: {tuple(sample.edge_index.shape)}")
        print(f"Sample edge_attr shape: {tuple(sample.edge_attr.shape)}")
        if task_spec.is_classification:
            print(f"Sample label: {int(sample.y.item())}")
            print(f"Sample raw id: {int(sample.target_raw.item())}")
        elif task_spec.num_outputs > 1:
            print(f"Sample scaled target (x, y, z): {sample.y.tolist()}")
            print(f"Sample raw target ({target_transform.unit}): {sample.target_raw.tolist()}")
        else:
            print(f"Sample scaled target: {float(sample.y.item()):.5f}")
            print(f"Sample raw target ({target_transform.unit}): {float(sample.target_raw.item()):.3f}")
        full_data.close()

    model = TPCGATModel(
        in_channels=input_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        edge_dim=5,
        event_feature_dim=event_feature_dim,
        head_hidden_channels=args.head_hidden_channels,
        dropout=args.dropout,
        activation=args.activation,
        task=args.task,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    criterion = task_spec.build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_epoch = 0
    if task_spec.is_classification:
        history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [], "val_roc_auc": []}
    else:
        history = {"train_loss": [], "val_loss": [], "train_scaled_mae": [], "val_scaled_mae": [], "val_mae": [], "val_rmse": []}
    history["lr"] = []
    if args.task == "regression_xyz":
        history.update({f"val_{key}": [] for key in VERTEX_METRIC_KEYS})
    history["lr_epoch"] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_metric, lrs = train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=args.grad_clip,
            epoch=epoch,
            log_interval=args.log_interval,
            task=args.task,
        )
        val_result = validate_epoch(
            model=model,
            device=device,
            val_loader=val_loader,
            criterion=criterion,
            target_abs_max=args.target_abs_max,
            target_mode=args.target_mode,
            task=args.task,
            target_transform=target_transform,
        )
        val_loss = val_result["loss"]

        if is_main_process:
            history["lr"].extend(lrs)
            history["lr_epoch"].extend((epoch - 1 + (np.arange(len(lrs)) + 1) / len(lrs)).tolist())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if task_spec.is_classification:
                roc_auc = val_result.get("roc_auc", float("nan"))
                print(
                    f"Epoch {epoch} Summary: "
                    f"Train Loss: {train_loss:.5f} | "
                    f"Train Accuracy: {train_metric:.5f} | "
                    f"Val Loss: {val_loss:.5f} | "
                    f"Val Accuracy: {val_result['accuracy']:.5f} | "
                    f"Val Precision: {val_result['precision']:.5f} | "
                    f"Val Recall: {val_result['recall']:.5f} | "
                    f"Val F1: {val_result['f1']:.5f} | "
                    f"Val ROC-AUC: {roc_auc:.5f}"
                )
                history["train_accuracy"].append(train_metric)
                history["val_accuracy"].append(val_result["accuracy"])
                history["val_precision"].append(val_result["precision"])
                history["val_recall"].append(val_result["recall"])
                history["val_f1"].append(val_result["f1"])
                history["val_roc_auc"].append(roc_auc)
            else:
                print(
                    f"Epoch {epoch} Summary: "
                    f"Train Loss: {train_loss:.5f} | "
                    f"Train Scaled MAE: {train_metric:.5f} | "
                    f"Val Loss: {val_loss:.5f} | "
                    f"Val Scaled MAE: {val_result['scaled_mae']:.5f} | "
                    f"Val MAE: {val_result['mae']:.5f} {target_transform.unit} | "
                    f"Val RMSE: {val_result['rmse']:.5f} {target_transform.unit}"
                )
                history["train_scaled_mae"].append(train_metric)
                history["val_scaled_mae"].append(val_result["scaled_mae"])
                history["val_mae"].append(val_result["mae"])
                history["val_rmse"].append(val_result["rmse"])
                if args.task == "regression_xyz":
                    for key in VERTEX_METRIC_KEYS:
                        history[f"val_{key}"].append(np.asarray(val_result[key]).tolist())
                    print(f"  Val MAE xyz [mm]: {val_result['mae_xyz']} | "
                          f"R68: {val_result['r68_mm']:.3f} mm | R95: {val_result['r95_mm']:.3f} mm")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                save_best_result(args.outdir, model, val_result, task_spec, target_transform)
                print(f"--> Saved new best model and evaluation metrics at Epoch {best_epoch}.")

    if is_main_process:
        total_time = (time.time() - start_time) / 60.0
        print()
        print(f"Training completed in {total_time:.2f} minutes.")
        print(f"Best model was found at Epoch {best_epoch} with Val Loss: {best_val_loss:.5f}")
        history.update(run_config)
        history.update(best_epoch=best_epoch, best_val_loss=best_val_loss,
                       input_channels=input_channels, event_feature_dim=event_feature_dim)
        if args.task == "regression_z":
            history["val_mae_mm"] = history["val_mae"]
            history["val_rmse_mm"] = history["val_rmse"]
        np.savez_compressed(os.path.join(args.outdir, "training_history.npz"),
                            **{key: np.asarray(value) for key, value in history.items() if value is not None})
        with open(os.path.join(args.outdir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
