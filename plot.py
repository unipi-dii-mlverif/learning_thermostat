from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt

def plot_columns(df, columns, output_file, time_learnt = float("-inf"), title=None, subtitle=None):
    plt.figure()
    plt.tight_layout()
    df.plot(x='time', y=columns)
    plt.axvline(time_learnt, ls = "--", color="gray")
    if title:
        plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, y=0.98, fontsize=10)
    plt.legend()
    plt.savefig(output_file, format='pdf')
    plt.close()

df = read_csv('build/stage2/outputs.csv')
df["c.heater"] = df.apply(lambda x: 1 if x['{Controller}.ControllerInstance.heater_on_out'] else 0, axis=1)
df["ml.heater"] = df.apply(lambda x: 1 if x['{ThermostatML}.ThermostatMLInstance.heater_on_out'] else 0, axis=1)
df["target"] = df.apply(lambda x: 25.5, axis=1)

time_learnt = (df[df['{ThermostatML}.ThermostatMLInstance.has_learnt'] == True].iloc[0])["time"] if len(df[df['{ThermostatML}.ThermostatMLInstance.has_learnt'] == True]) > 0 else float("-inf")

plot_columns(df, ['{Room}.RoomInstance.T_room_out', '{Plant}.PlantInstance.T_bair_out', '{Plant}.PlantInstance.T_heater_out', 'target'], "build/g_env.pdf", time_learnt, "Temperature over time", "T_des=25.5, LL = UL =2 ,H = 20, C = 30")
plot_columns(df, ["{ThermostatML}.ThermostatMLInstance.loss"], "build/g_loss.pdf", time_learnt)
plot_columns(df, ["c.heater", "ml.heater"], "build/g_act.pdf", time_learnt, "Heater activation over time", "T_des=25.5, LL = UL =2 ,H = 20, C = 30")

