#!/usr/bin/env python3
import pandas as pd
import sys

def merge_comparison_csvs(baseline_path, ml_path, output_path):
    """
    Merge baseline and ML CSV files for comparison.
    
    Args:
        baseline_path: Path to baseline CSV file
        ml_path: Path to ML CSV file
        output_path: Path to output merged CSV file
    """
    # Read the CSV files
    baseline_df = pd.read_csv(baseline_path)
    ml_df = pd.read_csv(ml_path)
    
    # Merge on time column
    merged_df = pd.merge(
        baseline_df[['time', '{Controller}.ControllerInstance.heater_on_out', '{Plant}.PlantInstance.T_bair_out']],
        ml_df[['time', '{ThermostatML}.ThermostatMLInstance.heater_on_out', '{Plant}.PlantInstance.T_bair_out']],
        on='time',
        suffixes=('_baseline', '_ml')
    )
    
    # Rename columns to desired names
    merged_df = merged_df.rename(columns={
        '{Controller}.ControllerInstance.heater_on_out': 'baseline_heater',
        '{Plant}.PlantInstance.T_bair_out_baseline': 'baseline_T_bair',
        '{ThermostatML}.ThermostatMLInstance.heater_on_out': 'ml_heater',
        '{Plant}.PlantInstance.T_bair_out_ml': 'ml_T_bair'
    })
    
    # Reorder columns
    merged_df = merged_df[['time', 'baseline_heater', 'baseline_T_bair', 'ml_heater', 'ml_T_bair']]
    
    # Save to output file
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_cmp.py <baseline_csv> <ml_csv> <output_csv>")
        sys.exit(1)
    
    baseline_csv = sys.argv[1]
    ml_csv = sys.argv[2]
    output_csv = sys.argv[3]
    
    merge_comparison_csvs(baseline_csv, ml_csv, output_csv)
