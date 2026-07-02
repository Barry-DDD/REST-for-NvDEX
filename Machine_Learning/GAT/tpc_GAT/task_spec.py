"""Task definitions for TPC graph learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGRESSION_TASK = "regression_z"
SUPPORTED_TASKS = (REGRESSION_TASK, "binary_id", "multiclass_id")
BINARY_ID_MAP = {1: 1, 21: 0, 22: 0, 23: 0, 24: 0}
MULTICLASS_ID_MAP = {1: 0, 21: 1, 22: 2, 23: 3, 24: 4}


@dataclass(frozen=True)
class TaskSpec:
    """Describe target handling, output shape, loss, and metrics for one task."""

    name: str
    task_type: str
    target_key: str
    num_outputs: int
    label_map: dict[int, int] | None = None
    positive_label: int | None = None

    @property
    def is_classification(self) -> bool:
        return self.task_type == "classification"

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def build_loss(self):
        """Return the default criterion for this task."""
        import torch.nn as nn

        if self.is_classification:
            return nn.CrossEntropyLoss()
        return nn.SmoothL1Loss()

    def map_raw_labels(self, raw_values):
        """Map raw HDF5 target values to train labels."""
        raw_values = np.asarray(raw_values).reshape(-1)
        if self.label_map is None:
            return raw_values.astype(np.float32, copy=False)

        labels = np.empty(raw_values.shape[0], dtype=np.int64)
        for i, value in enumerate(raw_values):
            raw_id = int(value)
            if raw_id not in self.label_map:
                allowed = ", ".join(str(k) for k in sorted(self.label_map))
                raise ValueError(f"Unsupported id {raw_id}. Supported ids for {self.name}: {allowed}.")
            labels[i] = int(self.label_map[raw_id])
        return labels


def get_task_spec(task: str) -> TaskSpec:
    """Return a task definition from its CLI name."""
    task = str(task)
    if task == "regression_z":
        return TaskSpec(
            name=task,
            task_type="regression",
            target_key="primary_origin_z",
            num_outputs=1,
        )
    if task == "binary_id":
        return TaskSpec(
            name=task,
            task_type="classification",
            target_key="id",
            num_outputs=2,
            label_map=BINARY_ID_MAP,
            positive_label=1,
        )
    if task == "multiclass_id":
        return TaskSpec(
            name=task,
            task_type="classification",
            target_key="id",
            num_outputs=5,
            label_map=MULTICLASS_ID_MAP,
        )
    raise ValueError(f"task must be one of: {', '.join(SUPPORTED_TASKS)}.")
