#!/usr/bin/env python3
"""
Aggregate statistics from simulation traces.
Computes average, std dev, variance, min, max for:
- T_bair
- T_heater
- Length of "heater on" periods
- Length of "heater off" periods
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path


def detect_run_type(df):
    """Detect if this is an 'ml' or 'baseline' run based on columns"""
    if any('ThermostatML' in col for col in df.columns):
        return 'ml'
    elif any('Controller' in col for col in df.columns):
        return 'baseline'
    else:
        raise ValueError("Cannot detect run type from columns")


def get_column_prefix(run_type):
    """Get the appropriate column prefix based on run type"""
    if run_type == 'ml':
        return '{ThermostatML}.ThermostatMLInstance'
    else:  # baseline
        return '{Controller}.ControllerInstance'


def compute_period_lengths(heater_series, time_series):
    """Compute lengths of heater on/off periods"""
    on_periods = []
    off_periods = []
    
    # Convert heater_series to boolean
    heater_bool = heater_series.map(lambda x: str(x).lower() == 'true' if pd.notna(x) else False)
    
    if len(heater_bool) == 0:
        return on_periods, off_periods
    
    current_state = heater_bool.iloc[0]
    period_start = time_series.iloc[0]
    
    for i in range(1, len(heater_bool)):
        if heater_bool.iloc[i] != current_state:
            period_length = time_series.iloc[i] - period_start
            if current_state:  # was on
                on_periods.append(period_length)
            else:  # was off
                off_periods.append(period_length)
            
            current_state = heater_bool.iloc[i]
            period_start = time_series.iloc[i]
    
    # Handle the last period
    final_length = time_series.iloc[-1] - period_start
    if current_state:
        on_periods.append(final_length)
    else:
        off_periods.append(final_length)
    
    return on_periods, off_periods


def compute_statistics(values):
    """Compute average, std dev, variance, min, max"""
    if len(values) == 0:
        return {
            'average': None,
            'std_dev': None,
            'variance': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    arr = np.array(values)
    return {
        'average': float(np.mean(arr)),
        'std_dev': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'variance': float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr)
    }


def check_constraint_violations(on_periods, off_periods, H=30.0, C=20.0):
    """
    Check heating and cooling interval constraint violations.
    
    Args:
        on_periods: List of heater ON period durations
        off_periods: List of heater OFF period durations
        H: Maximum allowed heating interval (default 30s)
        C: Minimum allowed cooling interval (default 20s)
    
    Returns:
        Dictionary with violation counts and percentages
    """
    heating_violations = sum(1 for duration in on_periods if duration > H)
    cooling_violations = sum(1 for duration in off_periods if duration < C)
    
    total_heating = len(on_periods)
    total_cooling = len(off_periods)
    total_intervals = total_heating + total_cooling
    total_violations = heating_violations + cooling_violations
    
    return {
        'heating_violations': heating_violations,
        'cooling_violations': cooling_violations,
        'total_violations': total_violations,
        'heating_violation_pct': (heating_violations / total_heating * 100) if total_heating > 0 else 0.0,
        'cooling_violation_pct': (cooling_violations / total_cooling * 100) if total_cooling > 0 else 0.0,
        'total_violation_pct': (total_violations / total_intervals * 100) if total_intervals > 0 else 0.0
    }


def analyze_trace(csv_path, H=30.0, C=20.0):
    """Analyze a single simulation trace"""
    df = pd.read_csv(csv_path)
    run_type = detect_run_type(df)
    prefix = get_column_prefix(run_type)
    
    # Extract temperature setting from path (e.g., build2/ml/32/outputs.csv -> 32)
    path_parts = Path(csv_path).parts
    temp_setting = None
    for i, part in enumerate(path_parts):
        if part in ['ml', 'baseline'] and i + 1 < len(path_parts):
            try:
                temp_setting = float(path_parts[i + 1])
            except ValueError:
                pass
    
    results = {
        'file': str(csv_path),
        'run_type': run_type,
        'temp_setting': temp_setting
    }
    
    # T_bair statistics
    t_bair_col = f'{prefix}.T_bair_in'
    if t_bair_col in df.columns:
        results['T_bair'] = compute_statistics(df[t_bair_col].dropna())
    else:
        print(f"Warning: {t_bair_col} not found in {csv_path}")
    
    # T_heater statistics - use Plant output
    plant_t_heater = '{Plant}.PlantInstance.T_heater_out'
    if plant_t_heater in df.columns:
        results['T_heater'] = compute_statistics(df[plant_t_heater].dropna())
    else:
        print(f"Warning: {plant_t_heater} not found in {csv_path}")
    
    # Heater on/off period lengths
    heater_on_col = f'{prefix}.heater_on_out'
    if heater_on_col in df.columns and 'time' in df.columns:
        on_periods, off_periods = compute_period_lengths(df[heater_on_col], df['time'])
        
        results['heater_on_periods'] = compute_statistics(on_periods)
        results['heater_off_periods'] = compute_statistics(off_periods)
        
        # Check constraint violations
        violations = check_constraint_violations(on_periods, off_periods, H, C)
        results['violations'] = violations
    else:
        if heater_on_col not in df.columns:
            print(f"Warning: {heater_on_col} not found in {csv_path}")
        if 'time' not in df.columns:
            print(f"Warning: 'time' column not found in {csv_path}")
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate.py <csv_files...> [--H value] [--C value]")
        sys.exit(1)
    
    # Parse arguments
    csv_files = []
    H = 30.0  # Default maximum heating interval
    C = 20.0  # Default minimum cooling interval
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--H' and i + 1 < len(sys.argv):
            H = float(sys.argv[i + 1])
            i += 2
        elif arg == '--C' and i + 1 < len(sys.argv):
            C = float(sys.argv[i + 1])
            i += 2
        else:
            csv_files.append(arg)
            i += 1
    
    print(f"Constraint parameters: H={H} (max heating), C={C} (min cooling)")
    print()
    
    all_results = []
    
    for csv_file in csv_files:
        try:
            result = analyze_trace(csv_file, H=H, C=C)
            all_results.append(result)
            print(f"✓ Processed: {csv_file}")
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Write results to JSON
    output_file = 'build2/results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results written to {output_file}")
    
    # Also create a summary CSV
    summary_data = []
    for result in all_results:
        row = {
            'file': result['file'],
            'run_type': result['run_type'],
            'temp_setting': result.get('temp_setting')
        }
        
        for metric in ['T_bair', 'T_heater', 'heater_on_periods', 'heater_off_periods']:
            if metric in result:
                for stat in ['average', 'std_dev', 'variance', 'min', 'max', 'count']:
                    row[f'{metric}_{stat}'] = result[metric].get(stat)
        
        # Add violation statistics
        if 'violations' in result:
            violations = result['violations']
            row['heating_violations'] = violations['heating_violations']
            row['cooling_violations'] = violations['cooling_violations']
            row['total_violations'] = violations['total_violations']
            row['heating_violation_pct'] = violations['heating_violation_pct']
            row['cooling_violation_pct'] = violations['cooling_violation_pct']
            row['total_violation_pct'] = violations['total_violation_pct']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = 'build2/results.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary CSV written to {summary_csv}")


if __name__ == '__main__':
    main()
