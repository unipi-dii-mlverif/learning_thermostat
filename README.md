# Learning Thermostat

This project implements a ML-based thermostat control system using Functional Mock-up Units (FMUs) and co-simulation. The system consists of multiple components including a physical plant model and a room environment from the INTO-CPS's case-study.

The key feature is a neural network-based controller (ThermostatML) that learns optimal heating control strategies by observing a baseline controller, then takes over operation once trained. The system uses model swapping to seamlessly transition from the baseline controller to the learned model during runtime.

## Getting Started

To build and run the complete simulation pipeline:

```bash
make all
```

This will:
- Build all FMU components
- Run the simulation
- Generate performance graphs and reports

The trained model weights are saved to `/var/tmp/learning_thermostat/thermostat_nn_model.pt` and can be reused for inference in subsequent runs.
