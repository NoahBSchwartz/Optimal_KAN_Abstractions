"""
Copied from the FastKAN library implementation of a KAN (https://github.com/ZiyaoLi/fast-kan)
with an additional option to set use_layernorm = False (for simplicity
during verification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert (
                input_dim > 1
            ), "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2,
    ):
        """this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        """
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]  # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts,
        )  # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,  # Added this parameter
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.layers = nn.ModuleList(
            [
                FastKANLayer(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    use_layernorm=use_layernorm,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, use_layernorm=self.use_layernorm)
        return x


class AttentionWithFastKANTransform(nn.Module):

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
        use_layernorm: bool = True,  # Added this parameter
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.use_layernorm = use_layernorm
        self.linear_q = FastKANLayer(q_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_k = FastKANLayer(k_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_v = FastKANLayer(v_dim, total_dim, use_layernorm=use_layernorm)
        self.linear_o = FastKANLayer(total_dim, q_dim, use_layernorm=use_layernorm)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim, use_layernorm=use_layernorm)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:

        wq = (
            self.linear_q(q, use_layernorm=self.use_layernorm).view(
                *q.shape[:-1], 1, self.num_heads, -1
            )
            * self.norm
        )  # *q1hc
        wk = self.linear_k(k, use_layernorm=self.use_layernorm).view(
            *k.shape[:-2], 1, k.shape[-2], self.num_heads, -1
        )  # *1khc
        att = (wq * wk).sum(-1).softmax(-2)  # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v, use_layernorm=self.use_layernorm).view(
            *v.shape[:-2], 1, v.shape[-2], self.num_heads, -1
        )  # *1khc
        o = (att[..., None] * wv).sum(-3)  # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)  # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q, use_layernorm=self.use_layernorm)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o, use_layernorm=self.use_layernorm)
        return o


import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import (
    GUROBI_CMD,
    LpBinary,
    LpMaximize,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)
import scipy as sc
from sklearn.metrics import r2_score
import src.kan_verification_grapher as kan_verification_grapher
import ssl
import time
from torch import autograd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def train_mnist():
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data_mnist", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data_mnist", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = FastKAN([28 * 28, 3, 10], use_base_update=False, use_layernorm=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                l2_reg = sum(torch.norm(param, p=2) for param in model.parameters())
                loss = loss + 0.01 * l2_reg
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    lr=optimizer.param_groups[0]["lr"],
                )

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    return model


my_kan = FastKAN(
    [28 * 28, 3, 1], use_base_update=False, use_layernorm=False
)  # train_mnist()


def fit_line_through_points(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def find_all_bspline_segments(layer, input_index, output_index, max_segments):
    n_samples = 25
    # Fetch curve data ONCE
    x_tensor, y_tensor = layer.plot_curve(input_index, output_index, num_pts=n_samples)
    x_points, y_points = (
        x_tensor.detach().cpu().numpy(),
        y_tensor.detach().cpu().numpy(),
    )

    # 1. Precompute errors for all possible segments (ONCE)
    errors = np.full((n_samples, n_samples), np.inf)
    for start_idx in range(n_samples - 1):
        x_start = x_points[start_idx]
        y_start = y_points[start_idx]
        for end_idx in range(start_idx + 1, n_samples):
            x_end = x_points[end_idx]
            y_end = y_points[end_idx]
            slope, intercept = fit_line_through_points(x_start, y_start, x_end, y_end)
            segment_x = x_points[start_idx : end_idx + 1]
            segment_y = y_points[start_idx : end_idx + 1]
            predicted_y = slope * segment_x + intercept
            segment_error = np.max(np.abs(predicted_y - segment_y))
            errors[start_idx, end_idx] = segment_error

    # 2. Dynamic programming to find optimal segmentation (ONCE)
    dp_table = np.full((max_segments, n_samples), np.inf)
    backtrack = np.zeros((max_segments, n_samples), dtype=int)

    # Base Case: 1 segment
    for j in range(1, n_samples):
        dp_table[0, j] = errors[0, j]

    # Recurrence
    for i in range(1, max_segments):
        for j in range(i + 1, n_samples):
            # We try breaking at k, where k is between the start of this segment number and current point
            # k is the END of the previous segment
            for k in range(i, j):
                if dp_table[i - 1, k] == np.inf:
                    continue
                if errors[k, j] == np.inf:
                    continue

                curr_error = max(dp_table[i - 1, k], errors[k, j])
                if curr_error < dp_table[i, j]:
                    dp_table[i, j] = curr_error
                    backtrack[i, j] = k

    # Extract results for EVERY segment count from 1 to max_segments
    results = {}
    for seg_count in range(1, max_segments + 1):
        row_idx = seg_count - 1
        final_error = dp_table[row_idx, n_samples - 1]
        if final_error == np.inf:
            results[seg_count] = ([], np.inf)
            continue

        # Reconstruct segments for this specific count using the backtrack table
        segments = []
        curr_seg_end = n_samples - 1

        for i in range(row_idx, -1, -1):
            if i > 0:
                prev_seg_end = backtrack[i, curr_seg_end]
            else:
                prev_seg_end = 0

            x1, y1 = x_points[prev_seg_end], y_points[prev_seg_end]
            x2, y2 = x_points[curr_seg_end], y_points[curr_seg_end]
            slope, intercept = fit_line_through_points(x1, y1, x2, y2)
            segments.insert(0, (x1, x2, slope, intercept))

            curr_seg_end = prev_seg_end

        results[seg_count] = (segments, final_error)

    return results


def calculate_bspline_lipschitz_constant(
    layer: "FastKANLayer",
    input_idx: int,
    output_idx: int,
    num_pts: int = 1000,
    min_x: float = None,
    max_x: float = None,
) -> float:
    if min_x is None:
        min_x = layer.rbf.grid_min
    if max_x is None:
        max_x = layer.rbf.grid_max
    x_tensor, y_tensor = layer.plot_curve(input_idx, output_idx, num_pts=num_pts)
    x_np = x_tensor.detach().cpu().numpy()
    y_np = y_tensor.detach().cpu().numpy()
    mask = (x_np >= min_x) & (x_np <= max_x)
    x_np = x_np[mask]
    y_np = y_np[mask]

    dx = np.diff(x_np)
    dy = np.diff(y_np)
    nonzero_dx = dx != 0
    slopes = np.zeros_like(dx)
    slopes[nonzero_dx] = dy[nonzero_dx] / dx[nonzero_dx]
    lipschitz_constant = np.max(np.abs(slopes))
    return lipschitz_constant


def compute_dp_tables_lipschitz(kan_model, max_segments=15):
    error_tables = {}
    segments_tables = {}
    lipschitz_constants = {}

    for layer_idx, layer in enumerate(kan_model.layers):
        input_dim = layer.input_dim
        output_dim = layer.output_dim
        print(
            f"Analyzing Layer {layer_idx}: {input_dim} inputs -> {output_dim} outputs"
        )

        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                all_results = find_all_bspline_segments(
                    layer, input_idx, output_index=output_idx, max_segments=max_segments
                )

                error_table = {}
                segments_table = {}

                for k, (segs, err) in all_results.items():
                    segments_table[k] = segs
                    error_table[k] = err
                lipschitz_constant = calculate_bspline_lipschitz_constant(
                    layer=layer, input_idx=input_idx, output_idx=output_idx
                )

                error_tables[spline_key] = error_table
                segments_tables[spline_key] = segments_table
                lipschitz_constants[spline_key] = lipschitz_constant

    return error_tables, segments_tables, lipschitz_constants


def plot_lipschitz_comparison_by_layer(lipschitz_constants, kan_model, figsize=(14, 6)):
    num_layers = len(kan_model.layers)
    layer_data = {i: [] for i in range(num_layers)}

    for (layer_idx, input_idx, output_idx), lip_const in lipschitz_constants.items():
        if np.isfinite(lip_const):
            layer_data[layer_idx].append(lip_const)
    fig, ax2 = plt.subplots(1, 1, figsize=figsize)
    positions = []
    data_to_plot = []
    labels = []
    for layer_idx in range(num_layers):
        if layer_data[layer_idx]:
            positions.append(layer_idx)
            data_to_plot.append(layer_data[layer_idx])
            layer = kan_model.layers[layer_idx]
            labels.append(f"L{layer_idx}\n({layer.input_dim}→{layer.output_dim})")
    bp = ax2.boxplot(
        data_to_plot, positions=positions, patch_artist=True, labels=labels, widths=0.6
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Lipschitz Constant", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Lipschitz Constants Box Plot by Layer", fontsize=12, fontweight="bold"
    )
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


error_tables, segments_tables, lipschitz_constants = compute_dp_tables_lipschitz(
    my_kan, 30
)
# fig2 = plot_lipschitz_comparison_by_layer(lipschitz_constants, my_kan)
# plt.show()


def weight_dp_tables_lipschitz(
    kan_model, error_tables, segments_tables, lipschitz_constants
):
    node_sensitivities = {}
    num_layers = len(kan_model.layers)

    # Intialize final layer with all 1's
    final_layer = kan_model.layers[num_layers - 1]
    for out_idx in range(final_layer.output_dim):
        node_sensitivities[(num_layers, out_idx)] = 1.0

    # Calculate sensitivity for the inputs of a layer based on the output layer's sensitivities
    layer_idx = num_layers - 1
    while layer_idx >= 0:
        current_layer = kan_model.layers[layer_idx]
        for input_idx in range(current_layer.input_dim):
            sensitivity_sum = 0.0
            for output_idx in range(current_layer.output_dim):
                lipschitz_constant = lipschitz_constants.get(
                    (layer_idx, input_idx, output_idx), 0.0
                )
                child_node_sensitivity = node_sensitivities[(layer_idx + 1, output_idx)]
                sensitivity_sum += lipschitz_constant * child_node_sensitivity
            node_sensitivities[(layer_idx, input_idx)] = sensitivity_sum
        layer_idx -= 1

    # Weight the error table (just do sensitivities * table)
    weighted_error_tables = {}
    for layer_idx, layer in enumerate(kan_model.layers):
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                current_bspline_key = (layer_idx, input_idx, output_idx)
                # The error of a spline is added directly to its destination node (output_idx)
                sensitivity_of_current_output_node = node_sensitivities[
                    (layer_idx + 1, output_idx)
                ]
                weighted_table_for_bspline = {}
                for pair in error_tables[current_bspline_key].items():
                    num_segments = pair[0]
                    error = pair[1]
                    weighted_table_for_bspline[num_segments] = (
                        error * sensitivity_of_current_output_node
                    )
                weighted_error_tables[current_bspline_key] = weighted_table_for_bspline
    return weighted_error_tables


def plot_error_comparison_by_layer(
    error_tables, weighted_error_tables, kan_model, max_segments=15, figsize=(14, 8)
):
    num_layers = len(kan_model.layers)
    sample_splines = []
    for layer_idx in range(num_layers):
        layer = kan_model.layers[layer_idx]
        # Pick ALL splines in this layer
        for input_idx in range(layer.input_dim):
            for output_idx in range(layer.output_dim):
                spline_key = (layer_idx, input_idx, output_idx)
                if spline_key in error_tables and spline_key in weighted_error_tables:
                    sample_splines.append(spline_key)
                else:
                    print(f"Warning: Spline {spline_key} not found in tables")
    if not sample_splines:
        print("No valid splines found to plot")
        return None
    num_splines = len(sample_splines)
    cols = min(4, num_splines)
    rows = (num_splines + cols - 1) // cols
    if figsize is None:
        figsize = (5 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16), squeeze=False)
    axes = axes.flatten()
    for idx, spline_key in enumerate(sample_splines):
        layer_idx, input_idx, output_idx = spline_key
        layer = kan_model.layers[layer_idx]
        original_errors = error_tables[spline_key]
        weighted_errors = weighted_error_tables[spline_key]
        num_segments_list = sorted(
            [k for k in original_errors.keys() if k <= max_segments]
        )
        original_vals = [
            original_errors[k]
            for k in num_segments_list
            if np.isfinite(original_errors[k])
        ]
        weighted_vals = [
            weighted_errors[k]
            for k in num_segments_list
            if np.isfinite(weighted_errors[k])
        ]
        min_len = min(len(num_segments_list), len(original_vals), len(weighted_vals))
        num_segments_list = num_segments_list[:min_len]
        original_vals = original_vals[:min_len]
        weighted_vals = weighted_vals[:min_len]
        ax = axes[idx]
        ax.plot(
            num_segments_list,
            original_vals,
            "o-",
            label="Original Error",
            color="steelblue",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            num_segments_list,
            weighted_vals,
            "s-",
            label="Weighted Error",
            color="coral",
            linewidth=2,
            markersize=6,
        )
        ax.set_xlabel("Number of Segments", fontsize=10, fontweight="bold")
        ax.set_ylabel("Max Error", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Layer {layer_idx} ({layer.input_dim}→{layer.output_dim})\n"
            f"Spline ({input_idx}→{output_idx})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        if original_vals and weighted_vals and original_vals[-1] > 0:
            ratio = weighted_vals[-1] / original_vals[-1]
            ax.text(
                0.05,
                0.95,
                f"Weight Ratio: {ratio:.2f}x",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    plt.tight_layout()
    fig.suptitle(
        "Original vs Weighted Error by Layer", fontsize=13, fontweight="bold", y=1.02
    )
    return fig


weighted_error_tables = weight_dp_tables_lipschitz(
    my_kan, error_tables, segments_tables, lipschitz_constants
)
# fig = plot_error_comparison_by_layer(error_tables, weighted_error_tables, my_kan, max_segments=30)
# plt.show()


def solve_best_segment_allocation(weighted_error_tables, target_max_error):
    model = gp.Model()
    # Create binary variable x[spline_key, num_segments] for every choice and every spline
    x = {}
    binary_vars_for_splines = {}
    for spline_key, num_segment_options in weighted_error_tables.items():
        binary_vars_for_splines[spline_key] = []
        for pair in num_segment_options.items():
            k_segments = pair[0]
            error_val = pair[1]
            if np.isinf(error_val):
                continue
            x[(spline_key, k_segments)] = model.addVar(vtype=GRB.BINARY, obj=k_segments)
            binary_vars_for_splines[spline_key].append(x[(spline_key, k_segments)])
    model.update()

    # Make sure we can only pick 1 variable
    for pair in binary_vars_for_splines.items():
        spline_key = pair[0]
        variables = pair[1]
        model.addConstr(gp.quicksum(variables) == 1)
    # Make sure we're under the max error
    total_error_expr = gp.quicksum(
        weighted_error_tables[s_key][k] * x[(s_key, k)] for s_key, k in x.keys()
    )
    model.addConstr(total_error_expr <= target_max_error)

    # Minimize the total number of segments
    total_segments_expr = gp.quicksum(k * x[(s_key, k)] for s_key, k in x.keys())
    model.setObjective(total_segments_expr, GRB.MINIMIZE)

    # Optimize!
    model.optimize()
    optimal_allocation = {}
    if model.status == GRB.OPTIMAL:
        print(f"Optimization successful! Minimum Segments Required: {model.objVal}")
        for (spline_key, k_segments), variable in x.items():
            if variable.X > 0.5:
                optimal_allocation[spline_key] = k_segments
        return optimal_allocation, model.objVal
    elif model.status == GRB.INFEASIBLE:
        print(
            "Model is INFEASIBLE. It is impossible to achieve this low of an error with the available segment options."
        )
        return None, float("inf")
    else:
        print(f"Optimization ended with status code: {model.status}")
        return None, float("inf")


optimal_allocation, min_error = solve_best_segment_allocation(
    weighted_error_tables, 0.9
)


def get_spline_bounds(segments, x_min, x_max):
    y_min, y_max = np.inf, -np.inf
    for sx1, sx2, slope, intercept in segments:
        # Find the intersection of the segment domain and input domain
        overlap_start = max(sx1, x_min)
        overlap_end = min(sx2, x_max)
        # If they overlap, evaluate the line at the edges
        if overlap_start <= overlap_end:
            val_start = slope * overlap_start + intercept
            val_end = slope * overlap_end + intercept
            y_min = min(y_min, val_start, val_end)
            y_max = max(y_max, val_start, val_end)
    return y_min, y_max


def propagate_kan_intervals(
    kan_shape, segments_tables, optimal_allocation, input_lb, input_ub
):
    layer_bounds = []
    # Input layer
    current_lb = input_lb
    current_ub = input_ub
    layer_bounds.append((current_lb, current_ub))
    num_transitions = len(kan_shape) - 1

    # Propagate across all layers
    for layer_idx in range(num_transitions):
        in_dim = kan_shape[layer_idx]
        out_dim = kan_shape[layer_idx + 1]
        next_lb = np.zeros(out_dim)
        next_ub = np.zeros(out_dim)
        for dst in range(out_dim):
            # Min_sum = Sum(Min_parts), Max_sum = Sum(Max_parts)
            total_min = 0.0
            total_max = 0.0
            for src in range(in_dim):
                x_min = current_lb[src]
                x_max = current_ub[src]
                num_segs = optimal_allocation[(layer_idx, src, dst)]
                segs = segments_tables[(layer_idx, src, dst)][num_segs]

                s_min, s_max = get_spline_bounds(segs, x_min, x_max)
                total_min += s_min
                total_max += s_max
            next_lb[dst] = total_min
            next_ub[dst] = total_max
        layer_bounds.append((next_lb, next_ub))
        current_lb, current_ub = next_lb, next_ub
    return layer_bounds


def build_kan_milp_model(
    kan_shape, segments_tables, error_tables, optimal_allocation, x_min_vec, x_max_vec
):
    model = gp.Model()
    input_dim = kan_shape[0]
    all_layer_variables = []

    # Create Input Variables
    current_layer_range_variables = []
    for i in range(input_dim):
        input_range_variable = model.addVar(lb=x_min_vec[i], ub=x_max_vec[i])
        current_layer_range_variables.append(input_range_variable)
    all_layer_variables.append(current_layer_range_variables)

    # Build Hidden Layers Sequentially
    num_transitions = (
        len(kan_shape) - 1
    )  # If shape is [784, 32, 10], we have 2 transitions: 0->1 and 1->2
    for layer_idx in range(num_transitions):
        layer_input_dimension = kan_shape[layer_idx]
        layer_output_dimension = kan_shape[layer_idx + 1]

        # Initialize output variables for this layer (by creating our list of inputs to the next layer)
        next_layer_input_ranges = []
        for i in range(layer_output_dimension):
            next_layer_input_ranges.append(
                gp.LinExpr()
            )  # ex. layer2_neuron_i = layer1_neuron1 + layer1_neuron2 + ... (linear expression!)

        for src_idx in range(layer_input_dimension):
            for dst_idx in range(layer_output_dimension):
                # Extract PWL Points
                num_segs = optimal_allocation[(layer_idx, src_idx, dst_idx)]
                seg_data = segments_tables[(layer_idx, src_idx, dst_idx)][
                    num_segs
                ]  # table has all possible segment allocations, only extract the optimal one (num_segs)
                x_pts, y_pts = [], []
                for idx, (x1, x2, slope, intercept) in enumerate(seg_data):
                    x_pts.append(x1)
                    y_pts.append(slope * x1 + intercept)
                    if idx == len(seg_data) - 1:  # Add end point of last segment
                        x_pts.append(x2)
                        y_pts.append(slope * x2 + intercept)

                # Create output variable for each bspline (adding in error)
                approx_error = error_tables[(layer_idx, src_idx, dst_idx)][num_segs]
                error_var = model.addVar(lb=-approx_error, ub=approx_error)
                src_var = current_layer_range_variables[src_idx]
                result_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                model.addGenConstrPWL(
                    src_var, result_var, x_pts, y_pts
                )  # "The relationship between src_var and result_var must follow the path connected by the dots"
                next_layer_input_ranges[dst_idx] += (
                    result_var + error_var
                )  # fill in the output linear expressions we defined earlier (for each neuron in the next layer)

        # Create variables for next layer nodes
        next_layer_range_variables = []
        for j in range(layer_output_dimension):
            next_layer_range_variable = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            model.addConstr(next_layer_range_variable == next_layer_input_ranges[j])
            next_layer_range_variables.append(next_layer_range_variable)
        current_layer_range_variables = next_layer_range_variables

        all_layer_variables.append(current_layer_range_variables)

    model.update()
    return model, all_layer_variables


def solve_kan_interval_milp(
    kan_shape,
    segments_tables,
    error_tables,
    optimal_allocation,
    output_layer_mip_gap,
    x_min_vec,
    x_max_vec,
):
    precomputed_bounds = propagate_kan_intervals(
        kan_shape, segments_tables, optimal_allocation, x_min_vec, x_max_vec
    )
    model, all_layer_variables = build_kan_milp_model(
        kan_shape,
        segments_tables,
        error_tables,
        optimal_allocation,
        x_min_vec,
        x_max_vec,
    )

    # Apply the precomputed bounds
    for layer_idx, vars_in_layer in enumerate(all_layer_variables):
        lbs, ubs = precomputed_bounds[layer_idx]
        for neuron_idx, var in enumerate(vars_in_layer):
            var.lb = lbs[neuron_idx]
            var.ub = ubs[neuron_idx]

    model.update()
    # Solve only the final layer
    final_layer_vars = all_layer_variables[-1]
    min_bounds = np.zeros(len(final_layer_vars))
    max_bounds = np.zeros(len(final_layer_vars))
    print(f"Starting MILP verification for {len(final_layer_vars)} output neurons...")
    model.setParam("MIPGap", output_layer_mip_gap)
    for i, var in enumerate(final_layer_vars):
        print(f"  Optimizing Output {i}...")
        # Minimize
        model.setObjective(var, GRB.MINIMIZE)
        model.optimize()
        min_bounds[i] = model.ObjBound
        # Maximize
        model.setObjective(var, GRB.MAXIMIZE)
        model.optimize()
        max_bounds[i] = model.ObjBound
    return min_bounds, max_bounds


kan_shape = [my_kan.layers[0].input_dim]
for layer in my_kan.layers:
    kan_shape.append(layer.output_dim)
x_min_vec = np.full(kan_shape[0], -0.5)
x_max_vec = np.full(kan_shape[0], 0.5)

hidden_layer_mip_gap = 0.5
output_layer_mip_gap = 1e-4
min_outputs, max_outputs = solve_kan_interval_milp(
    kan_shape,
    segments_tables,
    error_tables,
    optimal_allocation,
    output_layer_mip_gap,
    x_min_vec,
    x_max_vec,
)

print("Minimum Output Bounds:", min_outputs)
print("Maximum Output Bounds:", max_outputs)
