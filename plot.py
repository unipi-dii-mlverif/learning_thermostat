from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

def plot_columns(df, columns, output_file):
    plt.figure()
    plt.tight_layout()
    df.plot(x='time', y=columns)
    plt.legend()
    plt.savefig(output_file, format='pdf')
    plt.close()

df = read_csv('build/stage2/outputs.csv')
df["c.heater"] = df.apply(lambda x: 1 if x['{Controller}.ControllerInstance.heater_on_out'] else 0, axis=1)
df["ml.heater"] = df.apply(lambda x: 1 if x['{ThermostatML}.ThermostatMLInstance.heater_on_out'] else 0, axis=1)

plot_columns(df, ['{Room}.RoomInstance.T_room_out', '{Plant}.PlantInstance.T_bair_out', '{Plant}.PlantInstance.T_heater_out', '{KalmanFilter}.KalmanFilterInstance.T_heater_out', '{KalmanFilter}.KalmanFilterInstance.T_bair_out'], "build/g_env.pdf")
plot_columns(df, ["{ThermostatML}.ThermostatMLInstance.loss"], "build/g_loss.pdf")
plot_columns(df, ["c.heater", "ml.heater"], "build/g_act.pdf")

