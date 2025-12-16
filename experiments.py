#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DSE_BASE_DIR = PROJECT_ROOT / "build" / "dse"
RL_TRAIN_BASE_DIR = PROJECT_ROOT / "build" / "rl_dse_train"
RL_EVAL_BASE_DIR = PROJECT_ROOT / "build" / "rl_dse_eval"
EXPERIMENTS_DIR = PROJECT_ROOT / "build" / "experiments"
TEMPERATURES_FILE = PROJECT_ROOT / "temperatures"

MM_DSE_RL_TEMPLATE = PROJECT_ROOT / "mm_dse_rl.template.json"
MM_DSE_EVAL_TEMPLATE = PROJECT_ROOT / "mm_dse_eval.template.json"

POLICY_PATH = "/var/tmp/learning_thermostat/thermostat_policy.pt"
THERMO_PREFIX = "{ThermostatML}.ThermostatMLInstance."

# colonne “di riferimento” nei CSV
ROOM_T_COL = "{Room}.RoomInstance.T_room_out"
PLANT_T_COL = "{Plant}.PlantInstance.T_bair_out"
PLANT_H_IN_COL = "{Plant}.PlantInstance.heater_on_in"
CTRL_H_OUT_COL = "{Controller}.ControllerInstance.heater_on_out"

# --------------------------------------------------------------------
# Experiments definition (reward priorities)
# --------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "CF",
        "description": "Comfort-first",
        "reward_alpha": 1.0,   # comfort weight
        "reward_beta": 0.02,   # energy weight
        "reward_lambda": 0.01, # commutation weight
        "bc_coef": 1.0,        # strong BC
    },
    {
        "name": "EF",
        "description": "Energy-first",
        "reward_alpha": 0.7,
        "reward_beta": 0.15,
        "reward_lambda": 0.02,
        "bc_coef": 0.5,
    },
    {
        "name": "SF",
        "description": "Anti-chattering (switching-first)",
        "reward_alpha": 0.7,
        "reward_beta": 0.05,
        "reward_lambda": 0.15,
        "bc_coef": 0.5,
    },
]

# --------------------------------------------------------------------
# Shell helper
# --------------------------------------------------------------------

def run_cmd(cmd: str):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")

# --------------------------------------------------------------------
# Templates patch (text, WITHOUT parsing JSON)
# --------------------------------------------------------------------

import re

def patch_template_file(template_path: Path, exp: dict, mode: str, save_to_disk: bool) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    text = template_path.read_text()

    # Trova il blocco "parameters": { ... }
    m = re.search(r'"parameters"\s*:\s*\{', text)
    if not m:
        raise RuntimeError(f'"parameters" object not found in {template_path}')
    start = m.end()  # posizione dopo '{'

    brace_level = 1
    i = start
    while i < len(text) and brace_level > 0:
        c = text[i]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
        i += 1
    end = i - 1  # '}' che chiude il parameters

    body = text[start:end]

    keyvals = {
        f"{THERMO_PREFIX}MODE": f'"{mode}"',
        f"{THERMO_PREFIX}SAVE_TO_DISK": "true" if save_to_disk else "false",
        f"{THERMO_PREFIX}USE_OFFLINE_RL": "false",
        f"{THERMO_PREFIX}model_load_path": f'"{POLICY_PATH}"',
        f"{THERMO_PREFIX}model_save_path": f'"{POLICY_PATH}"',
        f"{THERMO_PREFIX}reward_alpha": str(exp["reward_alpha"]),
        f"{THERMO_PREFIX}reward_beta": str(exp["reward_beta"]),
        # NB: model.py usa reward_lambda_switch
        f"{THERMO_PREFIX}reward_lambda_switch": str(exp["reward_lambda"]),
        f"{THERMO_PREFIX}BC_COEF": str(exp["bc_coef"]),
    }

    # Rimuovi eventuali definizioni già presenti
    for key in keyvals.keys():
        pattern = rf'\s*"{re.escape(key)}"\s*:\s*[^,\n]+,?\n?'
        body, _ = re.subn(pattern, "\n", body)

    body_stripped = body.rstrip()
    has_existing = bool(body_stripped.strip())
    needs_comma = has_existing and not body_stripped.strip().endswith(",")

    insert_lines = [f'    "{k}": {v}' for k, v in keyvals.items()]
    insert_block = ""
    if needs_comma:
        insert_block += ","
    insert_block += "\n" + ",\n".join(insert_lines) + "\n"

    new_body = body_stripped + insert_block
    new_text = text[:start] + new_body + text[end:]
    template_path.write_text(new_text)

    print(f"[experiments] Patched {template_path.name} for {exp['name']} (MODE={mode})")

def update_templates_for_experiment(exp: dict) -> None:
    patch_template_file(MM_DSE_RL_TEMPLATE, exp, mode="BC_RL", save_to_disk=True)
    patch_template_file(MM_DSE_EVAL_TEMPLATE, exp, mode="EVAL", save_to_disk=False)

# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------

def load_temperatures():
    temps = []
    with open(TEMPERATURES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                temps.append(line)
    return temps

def get_temp_series(df: pd.DataFrame):
    # Preferiamo la temperatura di stanza, altrimenti quella del plant
    if ROOM_T_COL in df.columns:
        return df[ROOM_T_COL].astype(float)
    elif PLANT_T_COL in df.columns:
        return df[PLANT_T_COL].astype(float)
    else:
        # fallback: prima colonna con 'T'
        candidates = [c for c in df.columns if "T_" in c or "T" in c]
        return df[candidates[0]].astype(float)

def get_heater_series(df: pd.DataFrame):
    # Vogliamo sempre il comando che arriva al Plant
    if PLANT_H_IN_COL in df.columns:
        return df[PLANT_H_IN_COL].astype(float)
    elif CTRL_H_OUT_COL in df.columns:
        # fallback, se non loggato heater_on_in
        return df[CTRL_H_OUT_COL].astype(float)
    else:
        # ultimo fallback: seconda colonna
        return df[df.columns[1]].astype(float)

def to01(s: pd.Series) -> pd.Series:
    return (s.astype(float) >= 0.5).astype(int)

def compute_metrics_for_scenario(
    base_csv: Path,
    rl_csv: Path,
    scenario_temp: float,
    experiment_name: str,
    scenario_name: str,
):
    df_base = pd.read_csv(base_csv)
    df_rl = pd.read_csv(rl_csv)

    # temperatura
    T_base = get_temp_series(df_base)
    T_rl = get_temp_series(df_rl)

    # comando applicato al plant
    h_base = to01(get_heater_series(df_base))
    h_rl = to01(get_heater_series(df_rl))

    temp_err_base = (T_base - scenario_temp).abs().mean()
    temp_err_rl = (T_rl - scenario_temp).abs().mean()

    heater_on_ratio_base = h_base.mean()
    heater_on_ratio_rl = h_rl.mean()

    # differenza tra baseline e RL (sul comando effettivo)
    diff_base_vs_rl = (h_base != h_rl).mean()

    # differenza tra RL e teacher nel run RL_EVAL (se teacher è loggato)
    if CTRL_H_OUT_COL in df_rl.columns:
        h_teacher = to01(df_rl[CTRL_H_OUT_COL])
        diff_rl_vs_teacher = (h_rl != h_teacher).mean()
    else:
        diff_rl_vs_teacher = float("nan")

    return {
        "experiment": experiment_name,
        "scenario": scenario_name,
        "T_desired": scenario_temp,
        "temp_err_base": float(temp_err_base),
        "temp_err_rl": float(temp_err_rl),
        "heater_on_ratio_base": float(heater_on_ratio_base),
        "heater_on_ratio_rl": float(heater_on_ratio_rl),
        "heater_diff_base_vs_rl": float(diff_base_vs_rl),
        "heater_diff_rl_vs_teacher": float(diff_rl_vs_teacher),
    }

# --------------------------------------------------------------------
# Single experiment run
# --------------------------------------------------------------------

def run_experiment(exp: dict, temperatures: list[str]):
    name = exp["name"]
    print("\n" + "=" * 40)
    print(f"=== Experiment {name}: {exp['description']}")
    print("=" * 40)

    update_templates_for_experiment(exp)

    # Pulizia solo delle directory RL e della policy
    for d in [RL_TRAIN_BASE_DIR, RL_EVAL_BASE_DIR]:
        if d.exists():
            print(f"[experiments] Removing {d}")
            run_cmd(f"rm -rf '{d}'")
    if os.path.exists(POLICY_PATH):
        print(f"[experiments] Removing old policy {POLICY_PATH}")
        os.remove(POLICY_PATH)

    # Train + Eval RL
    run_cmd("make rl_dse_train")
    run_cmd("make rl_dse_eval")

    rows = []
    for temp in temperatures:
        base_csv = DSE_BASE_DIR / temp / "outputs.csv"
        rl_csv = RL_EVAL_BASE_DIR / temp / "outputs.csv"
        if not base_csv.exists() or not rl_csv.exists():
            print(f"[experiments] WARNING: missing CSV for scenario {temp}, skipping.")
            continue

        try:
            scenario_temp = float(temp)
        except ValueError:
            print(f"[experiments] WARNING: cannot parse temperature from '{temp}', skipping.")
            continue

        row = compute_metrics_for_scenario(
            base_csv=base_csv,
            rl_csv=rl_csv,
            scenario_temp=scenario_temp,
            experiment_name=name,
            scenario_name=temp,
        )
        rows.append(row)

    if not rows:
        print(f"[experiments] No data for experiment {name}")
        return None

    df = pd.DataFrame(rows)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXPERIMENTS_DIR / f"rl_dse_analysis_{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[experiments] Saved per-scenario metrics to {out_csv}")

    # Riepilogo aggregato
    agg = df.mean(numeric_only=True)
    print("\n[experiments] Aggregate metrics:")
    print(agg)

    return df

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    temps = load_temperatures()
    print("Temperature scenarios:", temps)

    # Ricostruisci baseline una volta sola
    run_cmd("make clean")
    run_cmd("make")

    all_dfs = []
    for exp in EXPERIMENTS:
        df = run_experiment(exp, temps)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        all_csv = EXPERIMENTS_DIR / "rl_dse_all_experiments.csv"
        df_all.to_csv(all_csv, index=False)
        print(f"\n[experiments] Global metrics saved to {all_csv}")

if __name__ == "__main__":
    main()
