"""Edge-aware GATv2 models for TPC hit graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from task_spec import get_task_spec


def _normalize_activation(name):
    """Return an activation module from a CLI-friendly name."""
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("activation must be one of: relu, gelu, silu.")


class RegressionHead(nn.Module):
    """Regression head that returns one flat scalar per graph."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float, activation: str):
        super().__init__()
        mid_channels = max(int(hidden_channels) // 2, 16)
        self.layers = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), mid_channels),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(mid_channels, 1),
        )

    def forward(self, x):
        return self.layers(x).view(-1)


class ClassificationHead(nn.Module):
    """Classification head that returns logits per graph."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float, activation: str):
        super().__init__()
        mid_channels = max(int(hidden_channels) // 2, 16)
        self.layers = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), mid_channels),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(mid_channels, int(num_classes)),
        )

    def forward(self, x):
        return self.layers(x)


class TPCGATBackbone(nn.Module):
    """
    Shared edge-aware GATv2 graph encoder.

    The model consumes PyG Batch objects from TPCGraphDataset. It uses
    GATv2Conv(edge_dim=5 or 6) layers, graph-level mean/max pooling, optional
    event-level features, and returns one event-level embedding per graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = 5,
        event_feature_dim: int = 0,
        head_hidden_channels: int = 128,
        dropout: float = 0.10,
        activation: str = "gelu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if heads <= 0:
            raise ValueError("heads must be positive.")

        self.event_feature_dim = int(event_feature_dim)
        self.activation = _normalize_activation(activation)
        self.input_proj = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # Self-loop fill: zeros for geometric/energy features, 1 for same_module (last).
        self_loop_fill = torch.zeros(int(edge_dim), dtype=torch.float32)
        self_loop_fill[-1] = 1.0
        for _ in range(int(num_layers)):
            self.convs.append(
                GATv2Conv(
                    in_channels=int(hidden_channels),
                    out_channels=int(hidden_channels),
                    heads=int(heads),
                    concat=False,
                    edge_dim=int(edge_dim),
                    dropout=float(dropout),
                    add_self_loops=True,
                    fill_value=self_loop_fill,
                )
            )
            self.norms.append(nn.LayerNorm(int(hidden_channels)))

        pooled_channels = int(hidden_channels) * 2
        if self.event_feature_dim > 0:
            self.event_mlp = nn.Sequential(
                nn.Linear(self.event_feature_dim, int(head_hidden_channels)),
                nn.LayerNorm(int(head_hidden_channels)),
                _normalize_activation(activation),
                nn.Dropout(float(dropout)),
            )
            pooled_channels += int(head_hidden_channels)
        else:
            self.event_mlp = None
        self.embedding_dim = pooled_channels

    def forward(self, data):
        """Return one graph embedding per event."""
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = self.activation(x)

        graph_features = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )

        if self.event_feature_dim > 0:
            if not hasattr(data, "event_features"):
                raise ValueError("Model expects event_features, but the batch does not contain them.")
            event_features = data.event_features.view(graph_features.size(0), -1)
            graph_features = torch.cat([graph_features, self.event_mlp(event_features)], dim=1)

        return graph_features


class TPCGATModel(nn.Module):
    """Task-aware GAT model with a shared backbone and task-specific head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = 5,
        event_feature_dim: int = 0,
        head_hidden_channels: int = 128,
        dropout: float = 0.10,
        activation: str = "gelu",
        task: str = "regression_z",
    ):
        super().__init__()
        self.task_spec = get_task_spec(task)
        self.backbone = TPCGATBackbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim,
            event_feature_dim=event_feature_dim,
            head_hidden_channels=head_hidden_channels,
            dropout=dropout,
            activation=activation,
        )
        self.embedding_dim = self.backbone.embedding_dim
        if self.task_spec.is_classification:
            self.head = ClassificationHead(
                self.embedding_dim,
                head_hidden_channels,
                self.task_spec.num_outputs,
                dropout,
                activation,
            )
        else:
            self.head = RegressionHead(self.embedding_dim, head_hidden_channels, dropout, activation)

    def forward(self, data):
        """Return task outputs for one PyG batch."""
        return self.head(self.backbone(data))


class TPCGATRegressor(TPCGATModel):
    """Backward-compatible regression model name."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("task", "regression_z")
        super().__init__(*args, **kwargs)
