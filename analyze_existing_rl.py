#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

DSE_BASE_DIR = PROJECT_ROOT / "build" / "dse"
RL_EVAL_DIR = PROJECT_ROOT / "build" / "rl_dse_eval"
EXPERIMENTS_DIR = PROJECT_ROOT / "build" / "experiments"
TEMPERATURES_FILE = PROJECT_ROOT / "temperatures"

# colonne “standard” nei CSV
ROOM_T_COL = "{Room}.RoomInstance.T_room_out"
PLANT_T_COL = "{Plant}.PlantInstance.T_bair_out"
PLANT_H_IN_COL = "{Plant}.PlantInstance.heater_on_in"
CTRL_H_OUT_COL = "{Controller}.ControllerInstance.heater_on_out"


def load_temperatures():
    temps = []
    with open(TEMPERATURES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                temps.append(line)
    return temps


def get_temp_series(df: pd.DataFrame):
    # preferisci temperatura di stanza; se manca, quella del plant
    if ROOM_T_COL in df.columns:
        return df[ROOM_T_COL].astype(float)
    if PLANT_T_COL in df.columns:
        return df[PLANT_T_COL].astype(float)

    # fallback: prima colonna che contiene "T"
    candidates = [c for c in df.columns if "T_" in c or "T" in c]
    if candidates:
        return df[candidates[0]].astype(float)
    # ultima spiaggia: seconda colonna
    return df[df.columns[1]].astype(float)


def get_heater_series(df: pd.DataFrame):
    # vogliamo il comando che arriva al Plant
    if PLANT_H_IN_COL in df.columns:
        return df[PLANT_H_IN_COL].astype(float)
    if CTRL_H_OUT_COL in df.columns:
        # in baseline coincidono; in RL_EVAL è solo teacher
        return df[CTRL_H_OUT_COL].astype(float)

    # fallback: seconda colonna
    return df[df.columns[1]].astype(float)


def to01(s: pd.Series) -> pd.Series:
    return (s.astype(float) >= 0.5).astype(int)


def compute_metrics_for_scenario(base_csv: Path, rl_csv: Path, scenario_temp: float, scenario_name: str):
    df_base = pd.read_csv(base_csv)
    df_rl = pd.read_csv(rl_csv)

    # temperatura
    T_base = get_temp_series(df_base)
    T_rl = get_temp_series(df_rl)

    # comando effettivo al plant
    h_base = to01(get_heater_series(df_base))
    h_rl = to01(get_heater_series(df_rl))

    temp_err_base = (T_base - scenario_temp).abs().mean()
    temp_err_rl = (T_rl - scenario_temp).abs().mean()

    heater_on_ratio_base = h_base.mean()
    heater_on_ratio_rl = h_rl.mean()

    # differenza tra baseline e RL sul comando al plant
    diff_base_vs_rl = (h_base != h_rl).mean()

    # differenza tra RL e teacher nel run RL_EVAL, se teacher è loggato
    if CTRL_H_OUT_COL in df_rl.columns:
        h_teacher = to01(df_rl[CTRL_H_OUT_COL])
        diff_rl_vs_teacher = (h_rl != h_teacher).mean()
    else:
        diff_rl_vs_teacher = float("nan")

    return {
        "scenario": scenario_name,
        "T_desired": scenario_temp,
        "temp_err_base": float(temp_err_base),
        "temp_err_rl": float(temp_err_rl),
        "heater_on_ratio_base": float(heater_on_ratio_base),
        "heater_on_ratio_rl": float(heater_on_ratio_rl),
        "heater_diff_base_vs_rl": float(diff_base_vs_rl),
        "heater_diff_rl_vs_teacher": float(diff_rl_vs_teacher),
    }


def main():
    if not DSE_BASE_DIR.exists():
        raise RuntimeError(f"Baseline dir not found: {DSE_BASE_DIR}")
    if not RL_EVAL_DIR.exists():
        raise RuntimeError(f"RL eval dir not found: {RL_EVAL_DIR}")

    temps = load_temperatures()
    print("Scenari (temperatures):", temps)

    rows = []
    for temp in temps:
        base_csv = DSE_BASE_DIR / temp / "outputs.csv"
        rl_csv = RL_EVAL_DIR / temp / "outputs.csv"
        if not base_csv.exists() or not rl_csv.exists():
            print(f"[WARN] missing CSV for scenario {temp}, skipping.")
            continue
        try:
            scenario_temp = float(temp)
        except ValueError:
            print(f"[WARN] cannot parse temperature from '{temp}', skipping.")
            continue

        row = compute_metrics_for_scenario(base_csv, rl_csv, scenario_temp, temp)
        rows.append(row)

    if not rows:
        print("No scenarios processed.")
        return

    df = pd.DataFrame(rows).sort_values("T_desired")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXPERIMENTS_DIR / "analysis_current_rl.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved per-scenario metrics to: {out_csv}")

    # riepilogo aggregato
    agg = df.mean(numeric_only=True)
    print("\n=== Aggregate metrics over all scenarios (baseline vs CURRENT RL) ===")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(agg)


if __name__ == "__main__":
    main()
