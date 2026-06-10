"""Training and validation helpers for TPC GAT regression."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


METADATA_KEYS = [
    "event_index",
    "n_hit",
    "n_hits",
    "n_hits_original",
    "total_energy",
    "total_energy_filtered",
    "n_primaries",
    "n_tracks",
    "primary_origin",
    "g4_total_deposited_energy",
    "g4_sensitive_volume_energy",
]


def reduce_value(value, average=True):
    """Reduce a scalar tensor across all distributed processes."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        torch.distributed.all_reduce(value)
        if average:
            value /= world_size
    return value


def _targets_from_batch(batch):
    """Return regression targets as a flat float tensor."""
    return batch.y.view(-1).float()


def _autocast_context(device):
    """Return a torch autocast context suitable for the current device."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def _scaler_enabled(scaler):
    """Return whether GradScaler is active."""
    return scaler is not None and scaler.is_enabled()


def train_epoch(model, device, train_loader, optimizer, criterion, scheduler, scaler, grad_clip, epoch, log_interval):
    """Run one training epoch."""
    model.train()
    losses = []
    mae_values = []
    lrs = []
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    for i, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)
        targets = _targets_from_batch(batch)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            outputs = model(batch)
            loss = criterion(outputs, targets)

        if _scaler_enabled(scaler):
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_stepped = scaler.get_scale() >= scale_before_step
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer_was_stepped = True

        if scheduler is not None and optimizer_was_stepped:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        loss_reduced = reduce_value(loss.detach(), average=True)
        losses.append(loss_reduced.item())

        mae = torch.mean(torch.abs(outputs.detach() - targets))
        mae_reduced = reduce_value(mae, average=True)
        mae_values.append(mae_reduced.item())

        if (i + 1) % log_interval == 0 and local_rank == 0:
            print(
                f"Train Epoch [{epoch}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"LR: {current_lr:.2E}, "
                f"Loss: {loss_reduced.item():.5f}, "
                f"Scaled MAE: {mae_reduced.item():.5f}"
            )

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_mae = float(np.mean(mae_values)) if mae_values else 0.0
    return mean_loss, mean_mae, lrs


def validate_epoch(model, device, val_loader, criterion, target_abs_max=845.0, target_mode="minus_one_one", deduplicate_by_event=True):
    """Run validation and gather scaled/raw predictions, labels, and metadata."""
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    raw_pred_list = []
    raw_label_list = []
    metadata_lists = {key: [] for key in METADATA_KEYS}

    world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1

    def gather_tensor(tensor):
        """Gather tensors from all distributed processes."""
        if world_size < 2:
            return tensor
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, tensor)
        return torch.cat(gather_list, dim=0)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            targets = _targets_from_batch(batch)

            with _autocast_context(device):
                outputs = model(batch)
                loss = criterion(outputs, targets)

            loss_reduced = reduce_value(loss.detach(), average=True)
            loss_list.append(loss_reduced.item())

            raw_outputs = inverse_scale_target_tensor(outputs.detach(), target_abs_max, target_mode)
            raw_targets = batch.target_raw.view(-1).float() if hasattr(batch, "target_raw") else inverse_scale_target_tensor(targets, target_abs_max, target_mode)

            pred_list.append(gather_tensor(outputs.detach()))
            label_list.append(gather_tensor(targets.detach()))
            raw_pred_list.append(gather_tensor(raw_outputs))
            raw_label_list.append(gather_tensor(raw_targets.detach()))

            for key in METADATA_KEYS:
                if hasattr(batch, key):
                    metadata_lists[key].append(gather_tensor(getattr(batch, key)))

    pred_tensor = torch.cat(pred_list, dim=0).cpu().numpy()
    label_tensor = torch.cat(label_list, dim=0).cpu().numpy()
    raw_pred_tensor = torch.cat(raw_pred_list, dim=0).cpu().numpy()
    raw_label_tensor = torch.cat(raw_label_list, dim=0).cpu().numpy()
    metadata = {key: torch.cat(values, dim=0).cpu().numpy() for key, values in metadata_lists.items() if values}

    if deduplicate_by_event and "event_index" in metadata:
        _, unique_pos = np.unique(metadata["event_index"].reshape(-1), return_index=True)
        unique_pos = np.sort(unique_pos)
        pred_tensor = pred_tensor[unique_pos]
        label_tensor = label_tensor[unique_pos]
        raw_pred_tensor = raw_pred_tensor[unique_pos]
        raw_label_tensor = raw_label_tensor[unique_pos]
        metadata = {key: value[unique_pos] for key, value in metadata.items()}

    residual = raw_pred_tensor - raw_label_tensor
    mae_mm = float(np.mean(np.abs(residual))) if residual.size else 0.0
    rmse_mm = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    scaled_residual = pred_tensor - label_tensor
    scaled_mae = float(np.mean(np.abs(scaled_residual))) if scaled_residual.size else 0.0

    return {
        "loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "preds_scaled": pred_tensor,
        "labels_scaled": label_tensor,
        "preds": raw_pred_tensor,
        "labels": raw_label_tensor,
        "mae_mm": mae_mm,
        "rmse_mm": rmse_mm,
        "scaled_mae": scaled_mae,
        "metadata": metadata,
    }


def inverse_scale_target_tensor(value, target_abs_max, target_mode):
    """Convert scaled model output back to primary_origin_z in mm."""
    if target_mode == "minus_one_one":
        return value * float(target_abs_max)
    if target_mode == "zero_one":
        return value * (2.0 * float(target_abs_max)) - float(target_abs_max)
    if target_mode == "none":
        return value
    raise ValueError("target_mode must be one of: 'minus_one_one', 'zero_one', 'none'.")
