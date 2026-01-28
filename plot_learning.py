#!/usr/bin/env python3
"""
Plot the learning reward over time from DSE stage2 outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        csv_file = "build/dse/34.0/stage2/outputs.csv"
    else:
        csv_file = sys.argv[1]
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Extract relevant columns
    time_col = 'time'
    loss_col = '{ThermostatML}.ThermostatMLInstance.loss'
    reward_col = '{ThermostatML}.ThermostatMLInstance.reward'
    has_learnt_col = '{ThermostatML}.ThermostatMLInstance.has_learnt'
    t_bair_col = '{ThermostatML}.ThermostatMLInstance.T_bair_in'
    heater_col = '{ThermostatML}.ThermostatMLInstance.heater_on_out'
    ll_col = '{Supervisor}.SupervisorInstance.LL_out'
    ul_col = '{Supervisor}.SupervisorInstance.UL_out'
    
    # Convert has_learnt to boolean
    df['has_learnt'] = df[has_learnt_col].map(lambda x: str(x).lower() == 'true')
    df['heater_on'] = df[heater_col].map(lambda x: str(x).lower() == 'true')
    
    # Get T_desired from path or use 34
    T_desired = 34.0
    
    # Find phase transitions
    bc_phase_end = df[df[loss_col] < 1.0]['time'].min() if (df[loss_col] < 1.0).any() else 0
    rl_phase_start = df[df['has_learnt']]['time'].min() if df['has_learnt'].any() else 0
    
    print(f"BC collection ends at: {bc_phase_end}")
    print(f"RL phase starts at: {rl_phase_start}")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Loss over time
    ax1.plot(df[time_col], df[loss_col], 'k-', linewidth=0.8)
    ax1.axvline(x=bc_phase_end, color='gray', linestyle='--', label='BC training start')
    ax1.axvline(x=rl_phase_start, color='gray', linestyle=':', label='RL phase start')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward over time (only during RL phase when rewards are computed)
    rl_data = df[df[time_col] >= rl_phase_start].copy()
    if len(rl_data) > 0:
        ax2.plot(rl_data[time_col], rl_data[reward_col], 'k-', linewidth=0.8)
        
        # Rolling average
        window = 20
        if len(rl_data) > window:
            rolling_avg = rl_data[reward_col].rolling(window=window, center=True).mean()
            ax2.plot(rl_data[time_col], rolling_avg, 'b-', linewidth=2, label=f'Rolling avg (w={window})')
            ax2.legend()
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward During RL Phase')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temperature trace
    # Get LL and UL values (they may vary over time)
    LL = df[ll_col].iloc[-1] if ll_col in df.columns else 5.0
    UL = df[ul_col].iloc[-1] if ul_col in df.columns else 0.0
    
    ax3.plot(df[time_col], df[t_bair_col], 'k-', linewidth=0.8, label='$T_{bair}$')
    ax3.axhline(y=T_desired, color='blue', linestyle='-', alpha=0.7, label=f'$T_{{desired}}$ = {T_desired}°C')
    ax3.axhline(y=T_desired - LL, color='green', linestyle='--', alpha=0.7, label=f'Lower bound ({T_desired - LL}°C)')
    ax3.axhline(y=T_desired + UL, color='red', linestyle='--', alpha=0.7, label=f'Upper bound ({T_desired + UL}°C)')
    ax3.axvline(x=rl_phase_start, color='gray', linestyle=':', label='RL phase start')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Air Temperature Trace')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latex/learning_reward.pdf', bbox_inches='tight')
    plt.savefig('latex/learning_reward.png', dpi=150, bbox_inches='tight')
    print("Saved plots to latex/learning_reward.pdf and latex/learning_reward.png")
    
    # Also export data for TikZ
    export_tikz_data(df, time_col, loss_col, reward_col, rl_phase_start)


def export_tikz_data(df, time_col, loss_col, reward_col, rl_phase_start):
    """Export sampled data for TikZ plots"""
    
    # Sample every 50th point to avoid too many coordinates
    step = 50
    sampled = df.iloc[::step].copy()
    
    # Loss data
    with open('latex/loss_data.tex', 'w') as f:
        f.write("% Loss data: time, loss\n")
        for _, row in sampled.iterrows():
            f.write(f"({row[time_col]:.0f}, {row[loss_col]:.4f})\n")
    
    # Reward data (RL phase only)
    rl_data = df[df[time_col] >= rl_phase_start].iloc[::10]  # Every 10th point
    with open('latex/reward_data.tex', 'w') as f:
        f.write("% Reward data: time, reward\n")
        for _, row in rl_data.iterrows():
            f.write(f"({row[time_col]:.0f}, {row[reward_col]:.4f})\n")
    
    print("Exported TikZ data to latex/loss_data.tex and latex/reward_data.tex")


if __name__ == '__main__':
    main()
