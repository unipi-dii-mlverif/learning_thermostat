from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

        # define data location\n",
df0 = read_csv('stage1/outputs.csv')
df0.to_csv("post/stage1.csv", index=False)
df1 = read_csv('transition/stage2/outputs.csv')
df1.to_csv("post/stage2.csv", index=False)
df0.fillna(0, inplace=True)
df1.fillna(0, inplace=True)
fig, ax = plt.subplots(1, sharex=False, sharey=False)
name1='{Room}.RoomInstance.T_room_out'
name2='{Plant}.PlantInstance.T_bair_out'
name3='{Plant}.PlantInstance.T_heater_out'
name4='{KalmanFilter}.KalmanFilterInstance.T_heater_out'
name5='{KalmanFilter}.KalmanFilterInstance.T_bair_out'

ax.title.set_text('Lerning Thermostat w/o swap')
ax.plot(df0['time'], df0[name1], label=name1)
ax.plot(df0['time'], df0[name2], label=name2)
ax.plot(df0['time'], df0[name3], label=name3)
ax.plot(df0['time'], df0[name4], label=name4)
ax.plot(df0['time'], df0[name5], label=name5)

ax.plot(df1['time'], df1[name1], label=name1)
ax.plot(df1['time'], df1[name2], label=name2)
ax.plot(df1['time'], df1[name3], label=name3)
ax.plot(df1['time'], df1[name4], label=name4)
ax.plot(df1['time'], df1[name5], label=name5)


plt.legend(loc="lower right")

#plt.yticks(np.arange(0.0, 2.5, 0.2))
#ax.set_xlim([0, 40])
ax.grid()
fig.tight_layout()
plt.rcParams['figure.figsize'] = [12, 10]
    #plt.rcParams['figure.dpi'] = 200 #e.g. is really fine, but slower\n",
#plt.show()
Path('post').mkdir(parents=True, exist_ok=True)
plt.savefig('post/result.png')
plt.savefig('post/result.pdf')
