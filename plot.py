from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import argparse
import os


def plot_columns(df, columns, output_file, time_learnt=float("-inf"), eval_time=None, t_des=None, ll=None, ul=None):
    """Plot specified columns with optional temperature reference lines."""
    plt.figure()
    plt.tight_layout()
    df.plot(x='time', y=columns, ax=plt.gca())
    plt.axvline(time_learnt, ls="--", color="gray")
    if eval_time is not None:
        plt.axvline(eval_time, ls="--", color="gray")
    
    # Add temperature reference lines if provided
    if t_des is not None:
        plt.axhline(t_des, ls=":", color="green", linewidth=1.5)
    if ll is not None:
        plt.axhline(ll, ls=":", color="blue", linewidth=1.5)
    if ul is not None:
        plt.axhline(ul, ls=":", color="red", linewidth=1.5)
    
    plt.legend()
    plt.savefig(output_file, format='pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot thermostat simulation results from CSV output.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='build/stage2/outputs.csv',
        help='Path to the CSV file with simulation outputs'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='build',
        help='Directory where output PDFs will be saved'
    )
    
    parser.add_argument(
        '--t-des',
        type=float,
        default=None,
        help='Desired temperature (T_desired) - will draw a green dotted line'
    )
    
    parser.add_argument(
        '--ll',
        type=float,
        default=None,
        help='Lower comfort limit (LL) - will draw a blue dotted line'
    )
    
    parser.add_argument(
        '--ul',
        type=float,
        default=None,
        help='Upper comfort limit (UL) - will draw a red dotted line'
    )
    
    parser.add_argument(
        '--eval-time',
        type=float,
        default=12000,
        help='Evaluation phase start time - will draw a grey dashed vertical line'
    )
    
    args = parser.parse_args()
    
    # Read CSV data
    df = read_csv(args.csv_path)
    
    # Create binary heater status columns
    df["c.heater"] = df.apply(
        lambda x: 1 if x['{Controller}.ControllerInstance.heater_on_out'] else 0, 
        axis=1
    )
    df["ml.heater"] = df.apply(
        lambda x: 1 if x['{ThermostatML}.ThermostatMLInstance.heater_on_out'] else 0, 
        axis=1
    )
    
    # Find when ML model finished learning
    learnt_rows = df[df['{ThermostatML}.ThermostatMLInstance.has_learnt'] == True]
    time_learnt = learnt_rows.iloc[0]["time"] if len(learnt_rows) > 0 else float("-inf")
    
    # Generate plots
    plot_columns(
        df,
        [
            '{Room}.RoomInstance.T_room_out',
            '{Plant}.PlantInstance.T_bair_out',
            '{Plant}.PlantInstance.T_heater_out',
            '{KalmanFilter}.KalmanFilterInstance.T_heater_out',
            '{KalmanFilter}.KalmanFilterInstance.T_bair_out'
        ],
        os.path.join(args.output_dir, "g_env.pdf"),
        time_learnt,
        args.eval_time,
        args.t_des,
        args.ll,
        args.ul
    )
    
    plot_columns(
        df,
        ["{ThermostatML}.ThermostatMLInstance.loss"],
        os.path.join(args.output_dir, "g_loss.pdf"),
        time_learnt,
        args.eval_time
    )
    
    plot_columns(
        df,
        ["c.heater", "ml.heater"],
        os.path.join(args.output_dir, "g_act.pdf"),
        time_learnt,
        args.eval_time
    )
    
    print(f"Plots saved to {args.output_dir}/")

    ##### PART 2 MIN/MAX/AVG HEATING TIME #############


if __name__ == "__main__":
    main()


