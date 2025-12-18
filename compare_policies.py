#!/usr/bin/env python3
"""
Compare BC vs RL policies for DSE temperature.
Computes KPIs: comfort RMSE, energy usage, switch count, late switches.
"""

import argparse
import json
import math
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame, t_desired: float, col_tbair: str, col_action: str) -> Dict[str, float]:
    """Compute performance metrics from outputs CSV."""
    tbair = df[col_tbair].to_numpy(dtype=np.float32)
    action = df[col_action].to_numpy()
    
    # Convert action to binary
    if action.dtype == np.bool_:
        action = action.astype(np.int64)
    else:
        action = (action.astype(np.float32) >= 0.5).astype(np.int64)
    
    # Comfort RMSE
    comfort_rmse = float(math.sqrt(np.mean((tbair - t_desired) ** 2)))
    
    # Energy fraction (% time heater ON)
    energy_frac = float(np.mean(action.astype(np.float32)))
    
    # Switch count
    switch_count = int(np.sum(action[1:] != action[:-1]))
    
    # Temperature violations
    ll = t_desired - 2.0  # Assuming standard LL
    ul = t_desired + 2.0  # Assuming standard UL
    violations = int(np.sum((tbair < ll) | (tbair > ul)))
    
    # Mean absolute temperature error
    mae = float(np.mean(np.abs(tbair - t_desired)))
    
    return {
        "comfort_rmse": comfort_rmse,
        "energy_frac": energy_frac,
        "switch_count": switch_count,
        "violations": violations,
        "mae": mae,
    }


def find_column(df: pd.DataFrame, patterns: list) -> str:
    """Find column matching any pattern (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for p in patterns:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    # Partial match
    for p in patterns:
        for c_lower, c_orig in cols_lower.items():
            if p.lower() in c_lower:
                return c_orig
    raise ValueError(f"Could not find column matching {patterns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, required=True, help="Target temperature")
    ap.add_argument("--bc-csv", required=True, help="BC policy outputs CSV")
    ap.add_argument("--rl-csv", required=True, help="RL policy outputs CSV")
    ap.add_argument("--out", required=True, help="Output report CSV")
    args = ap.parse_args()
    
    # Load CSVs
    df_bc = pd.read_csv(args.bc_csv)
    df_rl = pd.read_csv(args.rl_csv)
    
    # Detect columns
    try:
        col_tbair = find_column(df_bc, ["tbair", "t_bair", "plantinstance.tbair", "airtemp"])
        col_action_bc = find_column(df_bc, ["thermostatml", "mlheater", "heater_on_out"])
        col_action_rl = find_column(df_rl, ["thermostatml", "mlheater", "heater_on_out"])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    # Compute metrics
    metrics_bc = compute_metrics(df_bc, args.temp, col_tbair, col_action_bc)
    metrics_rl = compute_metrics(df_rl, args.temp, col_tbair, col_action_rl)
    
    # Compute improvements
    improvements = {}
    for key in metrics_bc:
        bc_val = metrics_bc[key]
        rl_val = metrics_rl[key]
        if bc_val != 0:
            improvement = ((bc_val - rl_val) / bc_val) * 100.0
        else:
            improvement = 0.0
        improvements[key] = improvement
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Temperature: {args.temp}°C")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'BC':<12} {'RL':<12} {'Δ %':<12}")
    print(f"{'-'*60}")
    print(f"{'Comfort RMSE':<20} {metrics_bc['comfort_rmse']:<12.3f} {metrics_rl['comfort_rmse']:<12.3f} {improvements['comfort_rmse']:<12.1f}")
    print(f"{'Energy Frac':<20} {metrics_bc['energy_frac']:<12.3f} {metrics_rl['energy_frac']:<12.3f} {improvements['energy_frac']:<12.1f}")
    print(f"{'Switch Count':<20} {metrics_bc['switch_count']:<12d} {metrics_rl['switch_count']:<12d} {improvements['switch_count']:<12.1f}")
    print(f"{'Violations':<20} {metrics_bc['violations']:<12d} {metrics_rl['violations']:<12d} {improvements['violations']:<12.1f}")
    print(f"{'MAE':<20} {metrics_bc['mae']:<12.3f} {metrics_rl['mae']:<12.3f} {improvements['mae']:<12.1f}")
    print(f"{'='*60}\n")
    
    # Write CSV report
    with open(args.out, 'w') as f:
        f.write("temp,metric,bc,rl,improvement_pct\n")
        for key in metrics_bc:
            f.write(f"{args.temp},{key},{metrics_bc[key]},{metrics_rl[key]},{improvements[key]}\n")
    
    # Write JSON for programmatic use
    json_path = args.out.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            "temp": args.temp,
            "bc": metrics_bc,
            "rl": metrics_rl,
            "improvements": improvements,
        }, f, indent=2)
    
    print(f"Report saved to: {args.out}")
    print(f"JSON saved to: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
