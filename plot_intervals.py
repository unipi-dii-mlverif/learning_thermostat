import torch
from torch import nn
import pickle
import os
from torchverif.interval_tensor.v2 import IntervalTensor
from torchverif.net_interval.v2 import *


# Define the model architecture (same as in model.py)
def create_model():
    model = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    return model

# Load the saved model
SAVE_DIR = "/var/tmp/learning_thermostat"
model_path = os.path.join(SAVE_DIR, "thermostat_nn_model.pt")

# Create model and load state dict
model = create_model()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

print(f"Model loaded from {model_path}")

# Perform one inference
# Input format: [T_bair_in, T_desired-LL_in, T_desired+UL_in, commutation_C_flag, commutation_H_flag]
# Example values:
T_bair = 24.5       # Current air temperature
LL = 2            # Lower limit offset
UL = 2            # Upper limit offset
T_desired = 25.5    # Desired temperature
C_flag = -1          # Commutation C flag (1 if time-commutation_time > C_in, else -1)
H_flag = -1         # Commutation H flag (1 if time-commutation_time > H_in, else -1)

input_tensor = torch.Tensor([
    T_bair,
    T_desired - LL,
    T_desired + UL,
    C_flag,
    H_flag
])

print(f"\nPerforming inference with input:")
print(f"  T_bair: {T_bair}")
print(f"  T_desired - LL: {T_desired - LL}")
print(f"  T_desired + UL: {T_desired + UL}")
print(f"  C_flag: {C_flag}")
print(f"  H_flag: {H_flag}")

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    heater_on = output[0] >= 0.5

print(f"\nModel output:")
print(f"  Raw output: {output[0].item():.4f}")
print(f"  Heater should be ON: {heater_on.item()}")


TEMPERATURE_INTERVALS = [
    #[17,18],
    [18, 18.5], [19, 19.5], [20, 20.5], [21, 21.5], [22, 22.5],
    [23, 23.5], [24, 24.5], [25, 25.5], [26, 26.5], [27, 27.5],
    [28, 28.5], [29, 29.5], [30, 30.5], [31, 31.5], #[32, 32.5],
    #[33, 33.5], [34, 34.5]
]

OUT_INTERVALS = []

for ti in TEMPERATURE_INTERVALS:
    lb = torch.Tensor([
        ti[0],
        T_desired - LL,
        T_desired + UL,
        C_flag,
        H_flag
    ])

    ub = torch.Tensor([
        ti[1],
        T_desired - LL,
        T_desired + UL,
        C_flag,
        H_flag
    ])

    i = IntervalTensor(lb, ub)

    o = model(i)
    print(bounds_from_v2_predictions(o)[0])

    OUT_INTERVALS.append([bounds_from_v2_predictions(o)[0][1], bounds_from_v2_predictions(o)[1][1]])

print(OUT_INTERVALS)

# Create area plot
import matplotlib.pyplot as plt
import numpy as np

# Calculate mean temperature for each interval
temperature_means = [np.mean(ti) for ti in TEMPERATURE_INTERVALS]

# Extract lower and upper bounds from OUT_INTERVALS
lower_bounds = [interval[0] for interval in OUT_INTERVALS]
upper_bounds = [interval[1] for interval in OUT_INTERVALS]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the area between lower and upper bounds
plt.fill_between(temperature_means, lower_bounds, upper_bounds, 
                 alpha=0.3, label='Output Interval Range', color='blue')

# Plot the bounds as lines
plt.plot(temperature_means, lower_bounds, 'b-', linewidth=2, label='Lower Bound')
plt.plot(temperature_means, upper_bounds, 'b-', linewidth=2, label='Upper Bound')

# Plot a dashed line at y = 0.5
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')

# Set custom x-axis ticks with interval labels
interval_labels = [f"[{ti[0]},{ti[1]}]" for ti in TEMPERATURE_INTERVALS]
plt.xticks(temperature_means, interval_labels, rotation=45, ha='right')

plt.xlabel('Temperature Intervals (°C)', fontsize=12)
plt.ylabel('Model Output', fontsize=12)
plt.title('Neural Network Output Intervals vs Temperature Intervals', fontsize=14)
plt.grid(True, alpha=0.3)
# plt.legend(loc='best')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(SAVE_DIR, 'temperature_output_intervals.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")

plt.show()

