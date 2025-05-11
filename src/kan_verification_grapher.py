import matplotlib.pyplot as plt
import scipy as sc
import math
import numpy as np
import os
from typing import Dict, Any

# ODE Visualization 
def system_eq_dis(cond_input, t_eval, time):
    def system_cont(t, x, d):
        dxdt = [x[1], x[0] - x[0]**3 - d*x[1]]
        return dxdt
    x0, y0, d = cond_input
    sol = sc.integrate.solve_ivp(system_cont, [0, time], (x0, y0), args=(d,), t_eval=t_eval)
    return sol.y.T, sol.t.T

def graph_ode_results(dataset, model, steps, t_eval, duration):
    for k in range(0, 2):
        plt.figure(figsize=[5, 4])
        sol, t_h = system_eq_dis(dataset[int(steps)*k, 1:4].detach().cpu(), t_eval.numpy(), duration)
        plt.plot(sol[:, 0], sol[:, 1], c='b', marker='x', label='gt')
        y_pred = model(dataset[int(steps)*k:int(steps)*(k+1)]).detach().cpu().numpy()
        plt.plot(y_pred[:, 0], y_pred[:, 1], c='r', marker='o', label='pred')
        plt.grid()
        x0, y0, d = dataset[int(steps)*k, 1:4].detach().numpy()
        plt.title(f"Predictions $x_0={x0:.3f}, y_0={y0:.3f}, d={d:.3f}$")
        plt.xlabel(r"$x_t$")
        plt.ylabel(r"$y_t$")
        plt.legend()
        plt.show() 


# Vanilla Visualization
def plot_kan(
    kan_model,
    all_segments,
    figsize=(15, 10),
    min_x=-5.0,
    max_x=5.0,
    num_pts=500,
):
    total_curves = len(all_segments)
    cols = math.ceil(math.sqrt(total_curves))
    rows = math.ceil(total_curves / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle('FastKAN Spline Curves Visualization', fontsize=16)
    axes = axes.flatten()
    for idx, segment_data in enumerate(all_segments):
        layer_idx = segment_data['layer_idx']
        input_index = segment_data['input_idx']
        output_index = segment_data['output_idx']
        segments = segment_data['segments']
        max_error = segment_data['max_error']
        ax = axes[idx]
        layer = kan_model.layers[layer_idx]
        # Plot original curve
        x_tensor, y_tensor = layer.plot_curve(
            input_index, output_index, 
            num_pts=num_pts
        )
        x_orig = x_tensor.detach().cpu().numpy()
        y_orig = y_tensor.detach().cpu().numpy()
        mask = (x_orig >= min_x) & (x_orig <= max_x)
        x_orig = x_orig[mask]
        y_orig = y_orig[mask]
        ax.plot(x_orig, y_orig, 'b-', linewidth=1.5, alpha=0.7, label='Original')
        # Plot segments if they exist
        if segments and max_error != np.inf:
            for x1, x2, slope, intercept in segments:
                x_seg = np.linspace(x1, x2, 50)
                y_seg = slope * x_seg + intercept
                ax.plot(x_seg, y_seg, 'r-', linewidth=2, alpha=0.7)
                ax.plot([x1], [slope * x1 + intercept], 'ro', markersize=3)
                ax.plot([x2], [slope * x2 + intercept], 'ro', markersize=3)
        
        error_text = f'Max err: {max_error:.3f}' if max_error != np.inf else 'Fitting failed'
        ax.text(0.05, 0.05, error_text, transform=ax.transAxes, 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        ax.set_title(f'Layer {layer_idx}, φ_{{{input_index},{output_index}}}', fontsize=10)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel(f'φ', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, alpha=0.3)
    for i in range(total_curves, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# Lipschitz Visualization
def print_dp_table(dp_table, num_splines, max_total_segments, step=20):
    print("\nDynamic Programming Table (Error values):")
    print("-" * 80)
    header = "Spline \\ Segments"
    for j in range(0, max_total_segments + 1, step):
        header += f" | {j:5d}"
    print(header)
    print("-" * len(header))
    for i in range(num_splines + 1):
        row = f"{i:3d}"
        for j in range(0, max_total_segments + 1, step):
            if np.isinf(dp_table[i, j]):
                row += " | inf  "
            else:
                row += f" | {dp_table[i, j]:.3f}"
        print(row)
    print("-" * 80)
    print("Note: 'inf' indicates that the error is infinite (impossible allocation)")
    print("Only the first 100 columns are shown (if applicable)")

def visualize_all_splines(kan_model, optimal_allocation, actual_segments):
    splines_by_layer = {}
    for spline_key in optimal_allocation.keys():
        layer_idx, input_idx, output_idx = spline_key
        if layer_idx not in splines_by_layer:
            splines_by_layer[layer_idx] = []
        splines_by_layer[layer_idx].append((input_idx, output_idx))
    figures = []
    # Create a separate figure for each layer
    for layer_idx, splines in splines_by_layer.items():
        layer = kan_model.layers[layer_idx]
        num_splines = len(splines)
        grid_size = math.ceil(math.sqrt(num_splines))
        rows = math.ceil(num_splines / grid_size)
        cols = min(grid_size, num_splines)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        fig.suptitle(f"Layer {layer_idx}: All Splines with Optimal Segment Allocation", fontsize=16)
        if num_splines > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        for i, (input_idx, output_idx) in enumerate(splines):
            if i >= len(axes_flat):
                break
            ax = axes_flat[i]
            spline_key = (layer_idx, input_idx, output_idx)
            x_tensor, y_tensor = layer.plot_curve(input_idx, output_idx, num_pts=1000)
            x_original = x_tensor.detach().cpu().numpy()
            y_original = y_tensor.detach().cpu().numpy()
            segments = actual_segments[spline_key]
            x_simplified = []
            y_simplified = []
            for x1, x2, slope, intercept in segments:
                segment_x = np.linspace(x1, x2, 50)
                segment_y = slope * segment_x + intercept
                x_simplified.extend(segment_x)
                y_simplified.extend(segment_y)
            ax.plot(x_original, y_original, 'b-', label='Original')
            ax.plot(x_simplified, y_simplified, 'r--', label=f'{optimal_allocation[spline_key]} segments')
            for x1, x2, _, _ in segments:
                ax.axvline(x=x1, color='g', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Input {input_idx} → Output {output_idx}')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, alpha=0.3)
        for j in range(num_splines, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)
    return figures


# MILP Visualization
def visualize_comparison_results(results: Dict[str, Any], output_dir: str):
    lipschitz_successful = "error" not in results["lipschitz"]
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    labels = ['Vanilla']
    vanilla_times = [results["vanilla"]["fit_time"]]
    if lipschitz_successful:
        labels.append('Lipschitz')
        lipschitz_times = [
            results["lipschitz"]["dp_time"],
            results["lipschitz"]["weighting_time"],
            results["lipschitz"]["allocation_time"]
        ]
        lipschitz_total = sum(lipschitz_times)
        vanilla_times.append(lipschitz_total)
        plt.bar(1, lipschitz_times[0], color='#ff8c00', label='DP Computation')
        plt.bar(1, lipschitz_times[1], bottom=lipschitz_times[0], color='#e74c3c', label='Error Weighting')
        plt.bar(1, lipschitz_times[2], bottom=sum(lipschitz_times[:2]), color='#9b59b6', label='Allocation')
    plt.bar(0, vanilla_times[0], color='#3498db', label='Vanilla Fitting')
    plt.title('Segment Fitting Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels)), labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 2)
    segments = [results["vanilla"]["total_segments"]]
    if lipschitz_successful:
        segments.append(results["lipschitz"]["total_segments"])
    plt.bar(range(len(segments)), segments, color=['#3498db', '#e74c3c'][:len(segments)])
    plt.title('Total Number of Segments')
    plt.ylabel('Number of Segments')
    plt.xticks(range(len(segments)), labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 3)
    verification_labels = ['Vanilla']
    verification_times = [results["verification"]["vanilla_mip_time"]]
    if lipschitz_successful and "lipschitz_mip_time" in results["verification"]:
        verification_labels.append('Lipschitz')
        verification_times.append(results["verification"]["lipschitz_mip_time"])
    plt.bar(range(len(verification_times)), verification_times, color=['#3498db', '#e74c3c'][:len(verification_times)])
    plt.title('MIP Verification Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(verification_times)), verification_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show figure before saving
    plt.show()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'output.png'), dpi=300)
    plt.close()

def detailed_metrics_visualization(results: Dict[str, Any], output_dir: str):
    lipschitz_successful = "error" not in results["lipschitz"]
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    vanilla_fitting_time = results["vanilla"]["fit_time"]
    vanilla_mip_time = results["verification"]["vanilla_mip_time"]
    if "vanilla_conversion_time" in results["verification"] and results["verification"]["vanilla_conversion_time"] > 0:
        vanilla_conversion_time = results["verification"]["vanilla_conversion_time"]
    else:
        vanilla_total = results["verification"]["vanilla_total_time"]
        vanilla_conversion_time = vanilla_total - vanilla_mip_time
    vanilla_times = [
        vanilla_fitting_time,
        vanilla_conversion_time,
        vanilla_mip_time
    ]
    labels = ['Vanilla']
    plt.bar(0, vanilla_times[0], color='#3498db', label='Fitting')
    plt.bar(0, vanilla_times[1], bottom=vanilla_times[0], color='#2ecc71', label='Conversion')
    plt.bar(0, vanilla_times[2], bottom=sum(vanilla_times[:2]), color='#f39c12', label='MIP')
    if lipschitz_successful and "lipschitz_mip_time" in results["verification"]:
        labels.append('Lipschitz')
        lipschitz_fitting_time = results["lipschitz"]["total_fit_time"]
        lipschitz_mip_time = results["verification"]["lipschitz_mip_time"]
        if "lipschitz_conversion_time" in results["verification"] and results["verification"]["lipschitz_conversion_time"] > 0:
            lipschitz_conversion_time = results["verification"]["lipschitz_conversion_time"]
        else:
            lipschitz_total = results["verification"]["lipschitz_total_time"]
            lipschitz_conversion_time = lipschitz_total - lipschitz_mip_time
        lipschitz_times = [
            lipschitz_fitting_time,
            lipschitz_conversion_time,
            lipschitz_mip_time
        ]
        plt.bar(1, lipschitz_times[0], color='#3498db')
        plt.bar(1, lipschitz_times[1], bottom=lipschitz_times[0], color='#2ecc71')
        plt.bar(1, lipschitz_times[2], bottom=sum(lipschitz_times[:2]), color='#f39c12')
    plt.title('End-to-End Time Breakdown')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels)), labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 2)
    if lipschitz_successful:
        lipschitz_times = [
            results["lipschitz"]["dp_time"],
            results["lipschitz"]["weighting_time"],
            results["lipschitz"]["allocation_time"]
        ]
        labels = ['DP Comp.', 'Error Weight.', 'Allocation']
        plt.bar(range(len(lipschitz_times)), lipschitz_times, color=['#ff8c00', '#e74c3c', '#9b59b6'])
        plt.title('Lipschitz Method Time Breakdown')
    else:
        plt.title('Lipschitz Method Not Available')
        plt.text(0.5, 0.5, 'Lipschitz method failed', 
                 horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(labels) if lipschitz_successful else 0), labels if lipschitz_successful else [])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplot(1, 3, 3)
    vanilla_results = results["verification"]["vanilla_results"]
    output_dim = len(vanilla_results)
    if lipschitz_successful and "lipschitz_results" in results["verification"]:
        lipschitz_results = results["verification"]["lipschitz_results"]
        x = np.arange(output_dim)
        width = 0.35
        vanilla_mins = [r[0] if r[0] is not None else 0 for r in vanilla_results]
        vanilla_maxs = [r[1] if r[1] is not None else 0 for r in vanilla_results]
        vanilla_ranges = [max_val - min_val for min_val, max_val in zip(vanilla_mins, vanilla_maxs)]
        lipschitz_mins = [r[0] if r[0] is not None else 0 for r in lipschitz_results]
        lipschitz_maxs = [r[1] if r[1] is not None else 0 for r in lipschitz_results]
        lipschitz_ranges = [max_val - min_val for min_val, max_val in zip(lipschitz_mins, lipschitz_maxs)]
        plt.bar(x - width/2, vanilla_ranges, width, label='Vanilla', color='#3498db')
        plt.bar(x + width/2, lipschitz_ranges, width, label='Lipschitz', color='#e74c3c')
    else:
        vanilla_mins = [r[0] if r[0] is not None else 0 for r in vanilla_results]
        vanilla_maxs = [r[1] if r[1] is not None else 0 for r in vanilla_results]
        vanilla_ranges = [max_val - min_val for min_val, max_val in zip(vanilla_mins, vanilla_maxs)]
        plt.bar(range(output_dim), vanilla_ranges, color='#3498db', label='Vanilla')
    plt.title('Output Range Width Comparison')
    plt.ylabel('Range Width')
    plt.xlabel('Output Dimension')
    plt.xticks(range(output_dim))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show figure before saving
    plt.show()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.png'), dpi=300)
    plt.close()