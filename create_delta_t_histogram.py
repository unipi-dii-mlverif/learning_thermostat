#!/usr/bin/env python3
"""
Create histogram data for temperature analysis.

Modes:
- 'excursion': ΔT = T - UL when T > UL (temperature excursions above limit)
- 'all': ΔT = T - T_desired for all samples (tracking error distribution)

This script aggregates data from all simulation traces (baseline and ML)
and computes the distribution.
"""

import csv
import pandas as pd
import numpy as np
import sys


def main(mode='excursion'):
    # Collect data for baseline and ML
    baseline_values = []
    ml_values = []

    if mode == 'excursion':
        # Collect delta_T = T - UL for T > UL
        # Process baseline traces
        for temp in range(30, 43):
            try:
                df = pd.read_csv(f'build2/baseline/{temp}/outputs.csv')

                # Get columns
                t_bair_col = '{Controller}.ControllerInstance.T_bair_in'
                ul_col = '{Supervisor}.SupervisorInstance.UL_out'

                if t_bair_col in df.columns and ul_col in df.columns:
                    # Compute absolute UL (desired temp + UL offset)
                    ul_absolute = temp + df[ul_col].iloc[0]

                    # Filter where T > UL
                    above_ul = df[df[t_bair_col] > ul_absolute]

                    # Compute delta T = T - UL
                    deltas = (above_ul[t_bair_col] - ul_absolute).values
                    baseline_values.extend(deltas)
            except Exception as e:
                print(f"Error processing baseline {temp}: {e}")

        # Process ML traces
        for temp in range(30, 43):
            try:
                df = pd.read_csv(f'build2/ml/{temp}/outputs.csv')

                # Get columns
                t_bair_col = '{ThermostatML}.ThermostatMLInstance.T_bair_in'
                ul_col = '{Supervisor}.SupervisorInstance.UL_out'

                if t_bair_col in df.columns and ul_col in df.columns:
                    # Compute absolute UL (desired temp + UL offset)
                    ul_absolute = temp + df[ul_col].iloc[0]

                    # Filter where T > UL
                    above_ul = df[df[t_bair_col] > ul_absolute]

                    # Compute delta T = T - UL
                    deltas = (above_ul[t_bair_col] - ul_absolute).values
                    ml_values.extend(deltas)
            except Exception as e:
                print(f"Error processing ML {temp}: {e}")

    elif mode == 'all':
        # Collect all temperature deltas ΔT = T - T_desired
        # Process baseline traces
        for temp in range(30, 43):
            try:
                df = pd.read_csv(f'build2/baseline/{temp}/outputs.csv')

                # Get columns
                t_bair_col = '{Controller}.ControllerInstance.T_bair_in'

                if t_bair_col in df.columns:
                    # Compute delta T = T - T_desired
                    deltas = df[t_bair_col].values - temp
                    baseline_values.extend(deltas)
            except Exception as e:
                print(f"Error processing baseline {temp}: {e}")

        # Process ML traces
        for temp in range(30, 43):
            try:
                df = pd.read_csv(f'build2/ml/{temp}/outputs.csv')

                # Get columns
                t_bair_col = '{ThermostatML}.ThermostatMLInstance.T_bair_in'

                if t_bair_col in df.columns:
                    # Compute delta T = T - T_desired
                    deltas = df[t_bair_col].values - temp
                    ml_values.extend(deltas)
            except Exception as e:
                print(f"Error processing ML {temp}: {e}")

    else:
        print(f"Unknown mode: {mode}")
        return

    print(f"Mode: {mode}")
    print(f"Baseline: {len(baseline_values)} samples")
    print(f"ML: {len(ml_values)} samples")

    # Create histogram bins based on mode
    if mode == 'excursion':
        bins = np.linspace(0, 5, 26)  # 0 to 5°C in 0.2°C increments
        output_file = 'paper/assets/plots/delta_t_histogram.csv'
    elif mode == 'all':
        bins = np.linspace(-15, 15, 31)  # -15 to 15°C in 1°C increments
        output_file = 'paper/assets/plots/temp_all_histogram.csv'
    
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute histograms
    baseline_hist, _ = np.histogram(baseline_values, bins=bins)
    ml_hist, _ = np.histogram(ml_values, bins=bins)

    # Normalize to percentages
    baseline_hist_pct = baseline_hist / len(baseline_values) * 100
    ml_hist_pct = ml_hist / len(ml_values) * 100

    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_center', 'baseline_pct', 'ml_pct'])
        for center, b_pct, m_pct in zip(bin_centers, baseline_hist_pct, ml_hist_pct):
            writer.writerow([f'{center:.2f}', f'{b_pct:.2f}', f'{m_pct:.2f}'])

    print(f"\nCreated {output_file}")
    print(f"Baseline stats: mean={np.mean(baseline_values):.3f}, std={np.std(baseline_values):.3f}, max={np.max(baseline_values):.3f}, min={np.min(baseline_values):.3f}")
    print(f"ML stats: mean={np.mean(ml_values):.3f}, std={np.std(ml_values):.3f}, max={np.max(ml_values):.3f}, min={np.min(ml_values):.3f}")


if __name__ == '__main__':
    # Get mode from command line argument if provided
    mode = sys.argv[1] if len(sys.argv) > 1 else 'excursion'
    
    if mode not in ['excursion', 'all']:
        print(f"Usage: {sys.argv[0]} [excursion|all]")
        print(f"  excursion: Histogram of ΔT = T - UL when T > UL")
        print(f"  all: Histogram of ΔT = T - T_desired for all samples")
        sys.exit(1)
    
    main(mode)
