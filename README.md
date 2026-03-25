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

## Repository Structure

### FMU Components (`FMU/`)

Each subdirectory contains the source files for one FMU component. The Makefile packages them into `.fmu` archives at build time.

| Component | Role |
|---|---|
| `Controller` | Baseline bang-bang thermostat controller |
| `KalmanFilter` | State estimator for room temperature |
| `Plant` | Physical heating plant model |
| `Room` | Room environment (from the INTO-CPS case study) |
| `Supervisor` | Orchestrates the learning/inference model swap |
| `ThermostatML` | Neural network controller that replaces the baseline after training |

### Simulation Configuration

| File | Purpose |
|---|---|
| `mm1.json` / `mm2.json` | Multi-model configs for phase 1 (learning with baseline) and phase 2 (inference with ML model) |
| `simulation-config.json` | Maestro co-simulation settings for the main run |
| `mm_cmp_baseline.json` / `mm_cmp_ml.json` | Multi-model configs for the baseline-vs-ML comparison |
| `simulation-config-cmp.json` | Maestro settings for the comparison run |
| `mm_dse.template.json` | DSE template; `%TEMP%` is substituted with each target outdoor temperature |
| `temperatures` | List of outdoor temperatures swept during DSE |

### Analysis Scripts

| Script | Description |
|---|---|
| `plot.py` | Generates PDF plots from simulation traces (environment, loss, actuator signals) |
| `plot_learning.py` | Plots reward evolution during the learning phase |
| `plot_intervals.py` | Plots heating/cooling interval length distributions |
| `create_delta_t_histogram.py` | Histogram of temperature delta values across runs |
| `check_intervals.py` | Verifies that heating/cooling intervals satisfy timing constraints |
| `analyze_cmp.py` | Computes comfort MSE and actuation divergence between baseline and ML |
| `merge_cmp.py` | Merges baseline and ML comparison CSVs into a single file |
| `aggregate.py` | Aggregates statistics (mean, std, min, max) across DSE simulation runs |
| `rl_train_from_result.py` | Offline RL/imitation-learning training from recorded simulation traces |

### Diagrams

| File | Description |
|---|---|
| `diagram.dot` / `diagram_mm1.dot` | Co-simulation architecture diagrams |
| `controller_automaton.dot` | Automaton diagram of the baseline controller logic |

### Build System

- `Makefile` — full pipeline: FMU packaging, Maestro co-simulation, DSE temperature sweep, comparison runs, and plot generation. Key targets: `all`, `dse`, `clean`.
- `aggregated_results.mak` — Makefile fragment for collecting and aggregating DSE results.
