import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from torchverif.interval_tensor.v2 import IntervalTensor
from torchverif.net_interval.v2 import bounds_from_v2_predictions

from . import classify as classify_mod
from .classify import _classify_cell, _classify_cell_temp_tsc, _classify_cell_tsc_deriv, _classify_cell_t_vs_tdes
from .config import (
    SAVE_DIR, T_desired, LL, UL, T_bair, T_derivative,
    time_since_comm, C_flag, H_flag,
)
from .model import make_state
from .tex_export import export_tex_interval_plot, export_tex_pgf


# ── Plot 0: output intervals vs temperature ───────────────────────────────────

def plot_temperature_intervals(model):
    T_MIN, T_MAX, T_STEP, T_WIDTH = 25.5, 37.0, 0.3, 0.3
    T_STARTS = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
    TEMPERATURE_INTERVALS = [[float(t), float(t + T_WIDTH)] for t in T_STARTS]

    out_intervals = []
    for ti in TEMPERATURE_INTERVALS:
        interval = IntervalTensor(
            make_state(ti[0], T_derivative),
            make_state(ti[1], T_derivative),
        )
        o = model(interval)
        bounds = bounds_from_v2_predictions(o)
        print(bounds[0])
        out_intervals.append([bounds[0][1], bounds[1][1]])

    print(out_intervals)

    tex_path = os.path.join(SAVE_DIR, 'temperature_output_intervals.tex')
    export_tex_interval_plot(TEMPERATURE_INTERVALS, out_intervals, tex_path)

    temperature_means = [np.mean(ti) for ti in TEMPERATURE_INTERVALS]
    lower_bounds = [iv[0] for iv in out_intervals]
    upper_bounds = [iv[1] for iv in out_intervals]

    plt.figure(figsize=(12, 6))
    plt.fill_between(temperature_means, lower_bounds, upper_bounds,
                     alpha=0.3, color='blue', label='Output Interval Range')
    plt.plot(temperature_means, lower_bounds, 'b-', linewidth=2, label='Lower Bound')
    plt.plot(temperature_means, upper_bounds, 'b-', linewidth=2, label='Upper Bound')
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')

    interval_labels = [f"[{ti[0]},{ti[1]}]" for ti in TEMPERATURE_INTERVALS]
    plt.xticks(temperature_means, interval_labels, rotation=45, ha='right')
    plt.xlabel('Temperature Intervals (°C)', fontsize=12)
    plt.ylabel('Model Output', fontsize=12)
    plt.title('Neural Network Output Intervals vs Temperature Intervals', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, 'temperature_output_intervals.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    plt.show()


# ── Plot 1: 2D region (Temperature vs T_derivative) ──────────────────────────
# Each cell is classified as:
#   0 = definitely OFF  (upper bound < 0.5)
#   1 = possibly ON     (lower bound < 0.5 but upper bound >= 0.5)
#   2 = definitely ON   (lower bound >= 0.5)

def plot_2d_region(model):
    classify_mod._pool_model = model  # inherited by forked workers

    T_STEP = 0.01
    DT_STEP = 0.01
    T_GRID = np.arange(24.0, 38.0, T_STEP)
    DT_GRID = np.arange(-1, 2, DT_STEP)

    result_matrix = np.zeros((len(DT_GRID), len(T_GRID)))

    tasks = [
        (j, i, float(t_lo), float(t_lo + T_STEP), float(dt_lo), float(dt_lo + DT_STEP))
        for j, t_lo in enumerate(T_GRID)
        for i, dt_lo in enumerate(DT_GRID)
    ]

    chunksize = max(1, len(tasks) // (32 * 4))
    with ProcessPoolExecutor(max_workers=32) as pool:
        for j, i, val in pool.map(_classify_cell, tasks, chunksize=chunksize):
            result_matrix[i, j] = val

    tex_path = os.path.join(SAVE_DIR, 'temp_vs_deriv_region.tex')
    export_tex_pgf(result_matrix, T_GRID, T_STEP, DT_GRID, DT_STEP,
                   r'Temperature $T$ ($^{\circ}$C)',
                   r'Temperature Derivative ($^{\circ}$C/s)',
                   'Heater Decision Regions', tex_path,
                   subtitle=f'LL={T_desired-LL}, UL={UL+T_desired}, $\\Delta t$={time_since_comm}\\,s $I_e = f(T)$')

    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = mcolors.ListedColormap(['#ff9999', '#ffee88', '#88cc88'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(T_GRID, DT_GRID, result_matrix, cmap=cmap, norm=norm, shading='nearest')

    cbar = plt.colorbar(mesh, ax=ax, ticks=[0.25, 1.0, 2.0])
    cbar.set_ticklabels(['Definitely OFF', 'Possibly ON', 'Definitely ON'])
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Temperature Derivative (°C/s)', fontsize=12)
    ax.set_title(
        'Heater Decision Regions — output ≥ 0.5\n'
        f'(T_desired={T_desired}, LL={LL}, UL={UL}, tsc={time_since_comm}s, '
        f'C_flag={C_flag:.2f}, H_flag={H_flag:.2f}, '
        f'integral_error=f(T))',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, 'temperature_derivative_region.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n2D region plot saved to {plot_path}")
    plt.show()


# ── Plot 2: 2D region (Temperature vs time-since-commutation) ────────────────

def plot_2d_temp_vs_tsc(model):
    classify_mod._pool_model = model

    DT_LO, DT_HI = 0.09, 0.3

    T_STEP = 0.02
    TSC_STEP = 0.05
    T_GRID   = np.arange(30, 35, T_STEP)
    TSC_GRID = np.arange(15, 40.0, TSC_STEP)

    result_matrix = np.zeros((len(TSC_GRID), len(T_GRID)))

    tasks = [
        (j, i, float(t_lo), float(t_lo + T_STEP), float(tsc_lo), float(tsc_lo + TSC_STEP), DT_LO, DT_HI)
        for j, t_lo   in enumerate(T_GRID)
        for i, tsc_lo in enumerate(TSC_GRID)
    ]

    chunksize = max(1, len(tasks) // (32 * 4))
    with ProcessPoolExecutor(max_workers=32) as pool:
        for j, i, val in pool.map(_classify_cell_temp_tsc, tasks, chunksize=chunksize):
            result_matrix[i, j] = val

    tex_path = os.path.join(SAVE_DIR, 'temp_vs_tsc_region.tex')
    export_tex_pgf(result_matrix, T_GRID, T_STEP, TSC_GRID, TSC_STEP,
                   r'Temperature ($^{\circ}$C)',
                   'Time since commutation (s)',
                   'Heater Decision Regions', tex_path,
                   subtitle=f'LL={T_desired-LL}, UL={T_desired+UL}, $\\dot{{T}}\\in[{DT_LO},{DT_HI}]$\\,$^{{\\circ}}$C/s')

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = mcolors.ListedColormap(['#ff9999', '#ffee88', '#88cc88'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(T_GRID, TSC_GRID, result_matrix, cmap=cmap, norm=norm, shading='nearest')
    cbar = plt.colorbar(mesh, ax=ax, ticks=[0.25, 1.0, 2.0])
    cbar.set_ticklabels(['Definitely OFF', 'Possibly ON', 'Definitely ON'])
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Time since commutation (s)', fontsize=12)
    ax.set_title(
        'Heater Decision Regions — output ≥ 0.5\n'
        f'(T_desired={T_desired}, LL={LL}, UL={UL}, '
        f'dT/dt\u2208[{DT_LO},{DT_HI}], integral_error=f(T))',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'temp_vs_tsc_region.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot 2 saved to {plot_path}")
    plt.show()


# ── Plot 3: 2D region (time-since-commutation vs T_derivative) ────────────────

def plot_2d_tsc_vs_deriv(model):
    classify_mod._pool_model = model

    TSC_STEP = 0.25
    DT_STEP  = 0.005
    TSC_GRID = np.arange(0.0, 40.0 + TSC_STEP / 2, TSC_STEP)
    DT_GRID  = np.arange(-0.4, 0.75 + DT_STEP / 2, DT_STEP)

    result_matrix = np.zeros((len(DT_GRID), len(TSC_GRID)))

    tasks = [
        (j, i, float(tsc_lo), float(tsc_lo + TSC_STEP), float(dt_lo), float(dt_lo + DT_STEP))
        for j, tsc_lo in enumerate(TSC_GRID)
        for i, dt_lo  in enumerate(DT_GRID)
    ]

    chunksize = max(1, len(tasks) // (32 * 4))
    with ProcessPoolExecutor(max_workers=32) as pool:
        for j, i, val in pool.map(_classify_cell_tsc_deriv, tasks, chunksize=chunksize):
            result_matrix[i, j] = val

    tex_path = os.path.join(SAVE_DIR, 'tsc_vs_deriv_region.tex')
    export_tex_pgf(result_matrix, TSC_GRID, TSC_STEP, DT_GRID, DT_STEP,
                   'Time since commutation (s)',
                   r'Temperature Derivative ($^{\circ}$C/s)',
                   'Heater Decision Regions', tex_path,
                   subtitle=f'LL={T_desired-LL}, UL={T_desired+UL}, $T_{{\\mathrm{{air}}}}$={T_bair}\\,$^{{\\circ}}$C')

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = mcolors.ListedColormap(['#ff9999', '#ffee88', '#88cc88'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(TSC_GRID, DT_GRID, result_matrix, cmap=cmap, norm=norm, shading='nearest')
    cbar = plt.colorbar(mesh, ax=ax, ticks=[0.25, 1.0, 2.0])
    cbar.set_ticklabels(['Definitely OFF', 'Possibly ON', 'Definitely ON'])
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Time since commutation (s)', fontsize=12)
    ax.set_ylabel('Temperature Derivative (°C/s)', fontsize=12)
    ax.set_title(
        'Heater Decision Regions — output ≥ 0.5\n'
        f'(T_desired={T_desired}, LL={LL}, UL={UL}, '
        f'T_bair={T_bair}, integral_error=f(T))',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'tsc_vs_deriv_region.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot 3 saved to {plot_path}")
    plt.show()


# ── Plot 4: 2D region (T_desired vs Temperature) ──────────────────────────────

def plot_2d_t_vs_tdes(model):
    classify_mod._pool_model = model

    T_STEP    = 0.05
    TDES_STEP = 0.05
    T_GRID    = np.arange(30.0, 40.0, T_STEP)
    TDES_GRID = np.arange(30.0, 40.0, TDES_STEP)

    result_matrix = np.zeros((len(T_GRID), len(TDES_GRID)))

    tasks = [
        (j, i, float(t_lo), float(t_lo + T_STEP), float(tdes_lo), float(tdes_lo + TDES_STEP))
        for j, tdes_lo in enumerate(TDES_GRID)
        for i, t_lo    in enumerate(T_GRID)
    ]

    chunksize = max(1, len(tasks) // (32 * 4))
    with ProcessPoolExecutor(max_workers=32) as pool:
        for j, i, val in pool.map(_classify_cell_t_vs_tdes, tasks, chunksize=chunksize):
            result_matrix[i, j] = val

    tex_path = os.path.join(SAVE_DIR, 't_vs_tdes_region.tex')
    export_tex_pgf(result_matrix, TDES_GRID, TDES_STEP, T_GRID, T_STEP,
                   r'Desired Temperature $T_{\mathrm{des}}$ ($^{\circ}$C)',
                   r'Temperature $T$ ($^{\circ}$C)',
                   'Heater Decision Regions', tex_path,
                   subtitle=f'LL={LL}, UL={UL}, $\\dot{{T}}=0$, $\\Delta t={time_since_comm}$\\,s')

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = mcolors.ListedColormap(['#ff9999', '#ffee88', '#88cc88'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(TDES_GRID, T_GRID, result_matrix, cmap=cmap, norm=norm, shading='nearest')
    cbar = plt.colorbar(mesh, ax=ax, ticks=[0.25, 1.0, 2.0])
    cbar.set_ticklabels(['Definitely OFF', 'Possibly ON', 'Definitely ON'])
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Desired Temperature (°C)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(
        'Heater Decision Regions — output \u2265 0.5\n'
        f'(LL={LL}, UL={UL}, dT/dt=0, tsc={time_since_comm}s, '
        f'C_flag={C_flag:.2f}, H_flag={H_flag:.2f})',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 't_vs_tdes_region.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot 4 saved to {plot_path}")
    plt.show()
