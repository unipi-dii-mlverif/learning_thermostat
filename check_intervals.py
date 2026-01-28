#!/usr/bin/env python3
"""
Check heating and cooling intervals from simulation output.

This script analyzes heating/cooling intervals from a CSV file and verifies
that they satisfy specified constraints:
- Heating intervals (heater on) should be <= H
- Cooling intervals (heater off) should be >= C
"""

import argparse
import pandas as pd
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check heating and cooling intervals from simulation output'
    )
    parser.add_argument('csv_file', type=str,
                        help='Path to the CSV output file')
    parser.add_argument('--T', type=float, required=True,
                        help='Start time threshold (only analyze data after this time)')
    parser.add_argument('--H', type=float, required=True,
                        help='Maximum allowed heating interval length')
    parser.add_argument('--C', type=float, required=True,
                        help='Minimum allowed cooling interval length')
    return parser.parse_args()


def parse_boolean(value):
    """Parse boolean values from CSV (handles 'true'/'false' strings)."""
    if isinstance(value, str):
        return value.lower() == 'true'
    return bool(value)


def find_intervals(df, heater_column, start_time):
    """
    Find all heating and cooling intervals after start_time.
    
    Returns:
        heating_intervals: list of (start_time, end_time, duration) tuples
        cooling_intervals: list of (start_time, end_time, duration) tuples
    """
    # Filter data after start_time
    df_filtered = df[df['time'] >= start_time].copy()
    
    if len(df_filtered) == 0:
        return [], []
    
    # Parse heater state
    df_filtered['heater_on'] = df_filtered[heater_column].apply(parse_boolean)
    
    heating_intervals = []
    cooling_intervals = []
    
    # Track current interval
    current_state = df_filtered.iloc[0]['heater_on']
    interval_start = df_filtered.iloc[0]['time']
    
    for idx in range(1, len(df_filtered)):
        row = df_filtered.iloc[idx]
        new_state = row['heater_on']
        current_time = row['time']
        
        # State change detected
        if new_state != current_state:
            interval_end = current_time
            duration = interval_end - interval_start
            
            if current_state:  # Was heating
                heating_intervals.append((interval_start, interval_end, duration))
            else:  # Was cooling
                cooling_intervals.append((interval_start, interval_end, duration))
            
            # Start new interval
            current_state = new_state
            interval_start = current_time
    
    # Handle the last interval (ongoing at end of simulation)
    final_time = df_filtered.iloc[-1]['time']
    duration = final_time - interval_start
    
    if current_state:
        heating_intervals.append((interval_start, final_time, duration))
    else:
        cooling_intervals.append((interval_start, final_time, duration))
    
    return heating_intervals, cooling_intervals


def check_constraints(heating_intervals, cooling_intervals, H, C):
    """
    Check if intervals satisfy constraints and return statistics.
    
    Returns:
        dict with statistics about violations
    """
    heating_violations = []
    cooling_violations = []
    
    for start, end, duration in heating_intervals:
        if duration > H:
            heating_violations.append((start, end, duration))
    
    for start, end, duration in cooling_intervals:
        if duration < C:
            cooling_violations.append((start, end, duration))
    
    stats = {
        'total_heating': len(heating_intervals),
        'total_cooling': len(cooling_intervals),
        'heating_violations': len(heating_violations),
        'cooling_violations': len(cooling_violations),
        'heating_violation_list': heating_violations,
        'cooling_violation_list': cooling_violations,
    }
    
    return stats


def main():
    """Main function."""
    args = parse_args()
    
    # Read CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if required column exists (try heater_on_out first, then ml_heater)
    heater_column = '{ThermostatML}.ThermostatMLInstance.heater_on_out'
    ml_heater_column = 'ml_heater'
    
    if heater_column in df.columns:
        column_to_use = heater_column
    elif ml_heater_column in df.columns:
        column_to_use = ml_heater_column
    else:
        print(f"Error: Neither '{heater_column}' nor '{ml_heater_column}' found in CSV", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using column: {column_to_use}")
    
    # Find intervals
    heating_intervals, cooling_intervals = find_intervals(df, column_to_use, args.T)
    
    print(f"Analysis starting from time T = {args.T}")
    print(f"Maximum heating interval (H) = {args.H}")
    print(f"Minimum cooling interval (C) = {args.C}")
    print()
    
    # Print all heating intervals
    print("=" * 80)
    print("HEATING INTERVALS (heater ON)")
    print("=" * 80)
    for i, (start, end, duration) in enumerate(heating_intervals, 1):
        status = "✓ OK" if duration <= args.H else "✗ VIOLATION"
        print(f"  #{i}: [{start:.1f}, {end:.1f}] duration = {duration:.1f}  {status}")
    print()
    
    # Print all cooling intervals
    print("=" * 80)
    print("COOLING INTERVALS (heater OFF)")
    print("=" * 80)
    for i, (start, end, duration) in enumerate(cooling_intervals, 1):
        status = "✓ OK" if duration >= args.C else "✗ VIOLATION"
        print(f"  #{i}: [{start:.1f}, {end:.1f}] duration = {duration:.1f}  {status}")
    print()
    
    # Check constraints and print statistics
    stats = check_constraints(heating_intervals, cooling_intervals, args.H, args.C)
    
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total heating intervals: {stats['total_heating']}")
    print(f"Total cooling intervals: {stats['total_cooling']}")
    print()
    print(f"Heating violations (duration > {args.H}): {stats['heating_violations']}")
    print(f"Cooling violations (duration < {args.C}): {stats['cooling_violations']}")
    print()
    
    # Detailed violation report
    if stats['heating_violations'] > 0:
        print("Heating violations details:")
        for start, end, duration in stats['heating_violation_list']:
            print(f"  - [{start:.1f}, {end:.1f}] duration = {duration:.1f} (exceeds H by {duration - args.H:.1f})")
        print()
    
    if stats['cooling_violations'] > 0:
        print("Cooling violations details:")
        for start, end, duration in stats['cooling_violation_list']:
            print(f"  - [{start:.1f}, {end:.1f}] duration = {duration:.1f} (below C by {args.C - duration:.1f})")
        print()
    
    # Summary
    total_violations = stats['heating_violations'] + stats['cooling_violations']
    total_intervals = stats['total_heating'] + stats['total_cooling']
    
    print("=" * 80)
    if total_violations == 0:
        print("✓ ALL CONSTRAINTS SATISFIED!")
    else:
        print(f"✗ CONSTRAINTS VIOLATED: {total_violations}/{total_intervals} intervals")
        success_rate = 100 * (1 - total_violations / total_intervals) if total_intervals > 0 else 0
        print(f"  Success rate: {success_rate:.1f}%")
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if total_violations == 0 else 1)


if __name__ == '__main__':
    main()
