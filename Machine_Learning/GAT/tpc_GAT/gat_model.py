"""Edge-aware GATv2 regressor for TPC hit graphs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


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


class TPCGATRegressor(nn.Module):
    """
    Lightweight edge-aware GATv2 regressor.

    The model consumes PyG Batch objects from TPCGraphDataset. It uses
    GATv2Conv(edge_dim=5) layers, graph-level mean/max pooling, optional
    event-level features, and a scalar regression head.
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

        self.regression_head = nn.Sequential(
            nn.Linear(pooled_channels, int(head_hidden_channels)),
            nn.LayerNorm(int(head_hidden_channels)),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_channels), max(int(head_hidden_channels) // 2, 16)),
            _normalize_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(head_hidden_channels) // 2, 16), 1),
        )

    def forward(self, data):
        """Return one scaled primary_origin_z prediction per event."""
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

        return self.regression_head(graph_features).view(-1)
