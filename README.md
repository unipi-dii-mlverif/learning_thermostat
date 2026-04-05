# Learning Thermostat

This project implements an ML-based thermostat control system using Functional Mock-up Units (FMUs) and Maestro co-simulation.

The core idea is a two-phase controller:
- a baseline thermostat controller runs first,
- a neural controller (`ThermostatML`) learns from the run,
- then the simulation transitions to the learned controller for inference.

Learned weights are saved to `/var/tmp/learning_thermostat/thermostat_nn_model.pt`.

## Prerequisites

- Java runtime (for Maestro)
- Maestro JAR (default used by `Makefile`): `~/Scaricati/maestro-4.0.2-jar-with-dependencies.jar`
- Python 3 with dependencies from `requirements.txt`

You can override the Maestro path at runtime:

```bash
make MAESTRO_JAR=/path/to/maestro-jar-with-dependencies.jar all
```

## Build Pipelines

This repository contains **two Makefiles** with different goals.

### 1) `Makefile` (main pipeline)

Run:

```bash
make all
```

What it does:
- Packages FMUs from `FMU/*` into `.fmu` archives.
- Imports and runs phase 1 (`mm1.json`) and phase 2 (`mm2.json`) with Maestro.
- Produces simulation outputs in `build/` and model weights in `/var/tmp/learning_thermostat/`.
- Generates default plots (`g_env.pdf`, `g_loss.pdf`, `g_act.pdf`).
- Runs DSE sweeps over temperatures listed in `temperatures` (`make dse`).
- Runs baseline-vs-ML comparison and merges traces into `build/cmp/result.csv`.

Useful targets:
- `make all` : full pipeline
- `make dse` : only DSE sweep
- `make clean` : remove generated artifacts

### 2) `aggregated_results.mak` (batch comparison + aggregation)

Run:

```bash
make -f aggregated_results.mak all
```

What it does:
- Sweeps desired temperature values from 30 to 42.
- For each value, runs both baseline and ML comparison scenarios.
- Stores raw outputs under `build2/baseline/<T>/` and `build2/ml/<T>/`.
- Calls `aggregate.py` to compute summary statistics and writes:
	- `build2/results.json`
	- `build2/results.csv`

Use this Makefile when you want aggregated cross-temperature statistics rather than the full training pipeline.

## Repository Structure

### FMU Components (`FMU/`)

Each subdirectory contains one FMU component source that is packaged into `.fmu` files at build time.

| Component | Role |
|---|---|
| `Controller` | Baseline thermostat controller |
| `KalmanFilter` | State estimator |
| `Plant` | Physical heating plant model |
| `Room` | Room environment model |
| `Supervisor` | Coordinates learning/inference phases |
| `ThermostatML` | Neural controller used after learning |

### Simulation Configuration

| File | Purpose |
|---|---|
| `mm1.json` | Phase 1 multi-model (baseline-guided learning) |
| `mm2.json` | Phase 2 multi-model (learned-controller inference) |
| `simulation-config.json` | Maestro settings for main simulation |
| `mm_cmp_baseline.json` / `mm_cmp_ml.json` | Comparison multi-models |
| `simulation-config-cmp.json` | Maestro settings for comparison runs |
| `mm_dse.template.json` | DSE template (`%TEMP%` substituted per run) |
| `temperatures` | Outdoor temperatures used by DSE |

### Analysis and Utility Scripts

| Script/Module | Description |
|---|---|
| `plot.py` | Generates simulation plots from CSV traces (`g_env.pdf`, `g_loss.pdf`, `g_act.pdf`, `g_reward.pdf`) |
| `plot_learning.py` | Plots learning metrics (loss, reward, temperature trace) from a run CSV |
| `plot_intervals` | Package for interval-based neural decision-region visualization (see below) |
| `create_delta_t_histogram.py` | Histogram analysis of temperature derivative data |
| `check_intervals.py` | Validates ON/OFF interval constraints (`--T`, `--H`, `--C`) |
| `merge_cmp.py` | Merges baseline and ML comparison CSVs |
| `analyze_cmp.py` | Quick comparison metrics on merged comparison output |
| `aggregate.py` | Aggregates statistics across many simulation outputs |
| `rl_train_from_result.py` | Offline BC/RL training from merged comparison traces |

### `plot_intervals`: what it does

`plot_intervals` is a Python package (directory), not a single script file.

Run it with:

```bash
python -m plot_intervals <mode>
```

Available modes:
- `0` : output interval bounds vs temperature intervals
- `1` : 2D decision regions (Temperature vs dT/dt)
- `2` : 2D decision regions (Temperature vs time-since-commutation)
- `3` : 2D decision regions (time-since-commutation vs dT/dt)
- `4` : 2D decision regions (Desired temperature vs Temperature)

For each grid cell, interval arithmetic is used to classify the neural output into:
- definitely OFF,
- possibly ON,
- definitely ON.

Outputs are saved under `/var/tmp/learning_thermostat/` as PNG and TeX artifacts.

### Diagrams

| File | Description |
|---|---|
| `diagram.dot` / `diagram_mm1.dot` | Co-simulation architecture diagrams |
| `controller_automaton.dot` | Baseline controller automaton |
