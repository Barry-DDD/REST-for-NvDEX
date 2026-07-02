"""Training and validation helpers for TPC GAT regression."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from task_spec import get_task_spec


METADATA_KEYS = [
    "event_index",
    "n_hit",
    "n_hits",
    "n_hits_original",
    "total_energy",
    "total_energy_filtered",
    "n_primaries",
    "n_tracks",
    "id",
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


def _targets_from_batch(batch, task_spec):
    """Return targets with the dtype expected by the task loss."""
    if task_spec.is_classification:
        return batch.y.view(-1).long()
    return batch.y.view(-1).float()


def _autocast_context(device):
    """Return a torch autocast context suitable for the current device."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def _scaler_enabled(scaler):
    """Return whether GradScaler is active."""
    return scaler is not None and scaler.is_enabled()


def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    scaler,
    grad_clip,
    epoch,
    log_interval,
    task="regression_z",
):
    """Run one training epoch."""
    task_spec = get_task_spec(task)
    model.train()
    losses = []
    metric_values = []
    lrs = []
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    for i, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)
        targets = _targets_from_batch(batch, task_spec)

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

        if task_spec.is_classification:
            metric = (torch.argmax(outputs.detach(), dim=1) == targets).float().mean()
            metric_name = "Accuracy"
        else:
            metric = torch.mean(torch.abs(outputs.detach() - targets))
            metric_name = "Scaled MAE"
        metric_reduced = reduce_value(metric, average=True)
        metric_values.append(metric_reduced.item())

        if (i + 1) % log_interval == 0 and local_rank == 0:
            print(
                f"Train Epoch [{epoch}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"LR: {current_lr:.2E}, "
                f"Loss: {loss_reduced.item():.5f}, "
                f"{metric_name}: {metric_reduced.item():.5f}"
            )

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_metric = float(np.mean(metric_values)) if metric_values else 0.0
    return mean_loss, mean_metric, lrs


def validate_epoch(
    model,
    device,
    val_loader,
    criterion,
    target_abs_max=845.0,
    target_mode="minus_one_one",
    deduplicate_by_event=True,
    task="regression_z",
):
    """Run validation and gather scaled/raw predictions, labels, and metadata."""
    task_spec = get_task_spec(task)
    model.eval()
    loss_list = []
    output_list = []
    label_list = []
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
            targets = _targets_from_batch(batch, task_spec)

            with _autocast_context(device):
                outputs = model(batch)
                loss = criterion(outputs, targets)

            loss_reduced = reduce_value(loss.detach(), average=True)
            loss_list.append(loss_reduced.item())

            output_list.append(gather_tensor(outputs.detach()))
            label_list.append(gather_tensor(targets.detach()))
            if hasattr(batch, "target_raw"):
                raw_label_list.append(gather_tensor(batch.target_raw.view(-1).detach()))

            for key in METADATA_KEYS:
                if hasattr(batch, key):
                    metadata_lists[key].append(gather_tensor(getattr(batch, key)))

    output_tensor = torch.cat(output_list, dim=0).cpu().numpy()
    label_tensor = torch.cat(label_list, dim=0).cpu().numpy()
    raw_label_tensor = torch.cat(raw_label_list, dim=0).cpu().numpy() if raw_label_list else label_tensor.copy()
    metadata = {key: torch.cat(values, dim=0).cpu().numpy() for key, values in metadata_lists.items() if values}

    if deduplicate_by_event and "event_index" in metadata:
        _, unique_pos = np.unique(metadata["event_index"].reshape(-1), return_index=True)
        unique_pos = np.sort(unique_pos)
        output_tensor = output_tensor[unique_pos]
        label_tensor = label_tensor[unique_pos]
        raw_label_tensor = raw_label_tensor[unique_pos]
        metadata = {key: value[unique_pos] for key, value in metadata.items()}

    if task_spec.is_classification:
        return _classification_result(loss_list, output_tensor, label_tensor, raw_label_tensor, metadata, task_spec)

    raw_pred_tensor = inverse_scale_target_tensor(torch.from_numpy(output_tensor), target_abs_max, target_mode).numpy()
    raw_label_tensor = raw_label_tensor.astype(np.float32, copy=False)
    residual = raw_pred_tensor - raw_label_tensor
    mae_mm = float(np.mean(np.abs(residual))) if residual.size else 0.0
    rmse_mm = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    scaled_residual = output_tensor - label_tensor
    scaled_mae = float(np.mean(np.abs(scaled_residual))) if scaled_residual.size else 0.0

    return {
        "loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "preds_scaled": output_tensor,
        "labels_scaled": label_tensor,
        "preds": raw_pred_tensor,
        "labels": raw_label_tensor,
        "mae_mm": mae_mm,
        "rmse_mm": rmse_mm,
        "scaled_mae": scaled_mae,
        "metadata": metadata,
    }


def _classification_result(loss_list, logits, labels, raw_labels, metadata, task_spec):
    """Build classification metrics and serializable outputs."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(raw_labels).reshape(-1)
    probabilities = _softmax_numpy(logits)
    predicted_class = np.argmax(probabilities, axis=1).astype(np.int64)
    confusion = _confusion_matrix(labels, predicted_class, task_spec.num_outputs)
    accuracy = float(np.mean(predicted_class == labels)) if labels.size else 0.0
    precision, recall, f1 = _classification_scores(confusion, task_spec)
    result = {
        "loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "logits": logits,
        "probabilities": probabilities,
        "predicted_class": predicted_class,
        "labels": labels,
        "raw_labels": raw_labels,
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "metadata": metadata,
    }
    if task_spec.num_outputs == 2:
        result["roc_auc"] = _binary_roc_auc(labels, probabilities[:, 1])
    return result


def _softmax_numpy(logits):
    """Compute stable softmax probabilities with NumPy."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _confusion_matrix(labels, predictions, num_classes):
    """Return a row=true, column=predicted confusion matrix."""
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for target, pred in zip(labels, predictions):
        if 0 <= int(target) < num_classes and 0 <= int(pred) < num_classes:
            matrix[int(target), int(pred)] += 1
    return matrix


def _classification_scores(confusion, task_spec):
    """Return binary positive-class or macro multiclass precision/recall/F1."""
    if task_spec.num_outputs == 2 and task_spec.positive_label is not None:
        cls = int(task_spec.positive_label)
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return float(precision), float(recall), float(f1)

    per_class = []
    for cls in range(confusion.shape[0]):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append((precision, recall, f1))
    return tuple(float(np.mean([item[i] for item in per_class])) for i in range(3))


def _binary_roc_auc(labels, scores):
    """Compute binary ROC-AUC from ranks; return nan when undefined."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(np.sum(ranks[positives]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def inverse_scale_target_tensor(value, target_abs_max, target_mode):
    """Convert scaled model output back to primary_origin_z in mm."""
    if target_mode == "minus_one_one":
        return value * float(target_abs_max)
    if target_mode == "zero_one":
        return value * (2.0 * float(target_abs_max)) - float(target_abs_max)
    if target_mode == "none":
        return value
    raise ValueError("target_mode must be one of: 'minus_one_one', 'zero_one', 'none'.")
