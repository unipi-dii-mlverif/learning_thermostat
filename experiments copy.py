#!/usr/bin/env python3
import os
import re
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

# --------------------------------------------------------------------
# Experiments definition (tune reward priorities)
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
        "reward_alpha": 0.5,
        "reward_beta": 0.15,
        "reward_lambda": 0.01,
        "bc_coef": 0.5,
    },
    {
        "name": "SF",
        "description": "Switching-first (anti-chattering)",
        "reward_alpha": 0.5,
        "reward_beta": 0.02,
        "reward_lambda": 0.15,
        "bc_coef": 0.5,
    },
]

# --------------------------------------------------------------------
# Utility: shell command wrapper
# --------------------------------------------------------------------

def run_cmd(cmd: str):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")

# --------------------------------------------------------------------
# Template patching (text-based, keeps %TEMP%)
# --------------------------------------------------------------------

def patch_template_file(template_path: Path, exp: dict, mode: str, save_to_disk: bool) -> None:
    """
    Patch mm_dse_* template as plain text, without JSON parsing,
    so that %TEMP% placeholders remain valid for the Makefile sed step.
    """

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    text = template_path.read_text()

    # 1) Locate "parameters": { ... } block
    m = re.search(r'"parameters"\s*:\s*\{', text)
    if not m:
        raise RuntimeError(f'"parameters" object not found in {template_path}')
    start = m.end()  # index just after '{'

    brace_level = 1
    i = start
    while i < len(text) and brace_level > 0:
        c = text[i]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
        i += 1
    end = i - 1  # index of the '}' closing the parameters object

    body = text[start:end]

    # 2) Build key/value pairs to inject/update
    keyvals = {
        f"{THERMO_PREFIX}MODE": f'"{mode}"',
        f"{THERMO_PREFIX}SAVE_TO_DISK": "true" if save_to_disk else "false",
        f"{THERMO_PREFIX}USE_OFFLINE_RL": "false",
        f"{THERMO_PREFIX}model_load_path": f'"{POLICY_PATH}"',
        f"{THERMO_PREFIX}model_save_path": f'"{POLICY_PATH}"',
        f"{THERMO_PREFIX}reward_alpha": str(exp["reward_alpha"]),
        f"{THERMO_PREFIX}reward_beta": str(exp["reward_beta"]),
        f"{THERMO_PREFIX}reward_lambda_switch": str(exp["reward_lambda"]),
        f"{THERMO_PREFIX}BC_COEF": str(exp["bc_coef"]),
    }

    # 3) Remove existing lines for those keys (if any)
    for key in keyvals.keys():
        pattern = rf'\s*"{re.escape(key)}"\s*:\s*[^,\n]+,?\n?'
        body, n = re.subn(pattern, "\n", body)

    body_stripped = body.rstrip()
    has_existing_entries = bool(body_stripped.strip())
    needs_comma = has_existing_entries and not body_stripped.strip().endswith(",")

    # 4) Build insertion block
    insert_lines = [f'    "{k}": {v}' for k, v in keyvals.items()]
    insert_block = ""
    if needs_comma:
        insert_block += ","
    insert_block += "\n" + ",\n".join(insert_lines) + "\n"

    new_body = body_stripped + insert_block

    # 5) Reassemble full file and write back
    new_text = text[:start] + new_body + text[end:]
    template_path.write_text(new_text)

    print(f"[experiments] Patched {template_path.name} for experiment {exp['name']} (MODE={mode})")

def update_templates_for_experiment(exp: dict) -> None:
    # RL training template: BC_RL mode, save policy to disk
    patch_template_file(MM_DSE_RL_TEMPLATE, exp, mode="BC_RL", save_to_disk=True)
    # Evaluation template: EVAL mode, no saving
    patch_template_file(MM_DSE_EVAL_TEMPLATE, exp, mode="EVAL", save_to_disk=False)

# --------------------------------------------------------------------
# Metrics computation
# --------------------------------------------------------------------

def load_temperatures():
    temps = []
    with open(TEMPERATURES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                temps.append(line)
    return temps

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)

    # Temperature column: prefer room temperature, otherwise controller T_bair_in
    if "{Room}.RoomInstance.T_room_out" in cols:
        T_col = "{Room}.RoomInstance.T_room_out"
    elif "{Controller}.ControllerInstance.T_bair_in" in cols:
        T_col = "{Controller}.ControllerInstance.T_bair_in"
    else:
        temp_candidates = [c for c in cols if "T_room" in c or "T_bair" in c]
        T_col = temp_candidates[0] if temp_candidates else cols[1]

    # Heater command column: prefer Controller.heater_on_out
    if "{Controller}.ControllerInstance.heater_on_out" in cols:
        H_col = "{Controller}.ControllerInstance.heater_on_out"
    elif "{Plant}.PlantInstance.heater_on_in" in cols:
        H_col = "{Plant}.PlantInstance.heater_on_in"
    else:
        heater_candidates = [
            c for c in cols
            if "heater" in c.lower() and ("on" in c.lower() or "cmd" in c.lower())
        ]
        H_col = heater_candidates[0] if heater_candidates else cols[1]

    return T_col, H_col

def compute_metrics_for_scenario(
    base_csv: Path,
    rl_csv: Path,
    scenario_temp: float,
    experiment_name: str,
    scenario_name: str,
) -> dict:
    df_base = pd.read_csv(base_csv)
    df_rl = pd.read_csv(rl_csv)

    T_col, H_col = detect_columns(df_base)

    T_base = df_base[T_col].astype(float)
    T_rl = df_rl[T_col].astype(float)
    h_base = df_base[H_col].astype(float)
    h_rl = df_rl[H_col].astype(float)

    temp_err_base = (T_base - scenario_temp).abs().mean()
    temp_err_rl = (T_rl - scenario_temp).abs().mean()

    heater_on_ratio_base = h_base.mean()
    heater_on_ratio_rl = h_rl.mean()
    diff_ratio = (h_base != h_rl).mean()

    return {
        "experiment": experiment_name,
        "scenario": scenario_name,
        "T_desired": scenario_temp,
        "temp_err_base": temp_err_base,
        "temp_err_rl": temp_err_rl,
        "heater_on_ratio_base": heater_on_ratio_base,
        "heater_on_ratio_rl": heater_on_ratio_rl,
        "heater_action_diff_ratio": diff_ratio,
    }

# --------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------

def run_experiment(exp: dict, temperatures: list[str]):
    name = exp["name"]
    print("\n" + "=" * 30)
    print(f"=== Experiment {name}: {exp['description']}")
    print("=" * 30)

    # Patch templates with correct MODE + reward weights
    update_templates_for_experiment(exp)

    # Clean RL-specific dirs and policy between experiments
    for d in [RL_TRAIN_BASE_DIR, RL_EVAL_BASE_DIR]:
        if d.exists():
            print(f"[experiments] Removing directory {d}")
            subprocess.run(f"rm -rf '{d}'", shell=True, check=False)
    if os.path.exists(POLICY_PATH):
        print(f"[experiments] Removing previous policy at {POLICY_PATH}")
        os.remove(POLICY_PATH)

    # Run RL DSE training and evaluation
    run_cmd("make rl_dse_train")
    run_cmd("make rl_dse_eval")

    # Collect metrics for all scenarios
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
            # se per qualche motivo la cartella non è numerica, salta
            print(f"[experiments] WARNING: cannot parse scenario temperature from '{temp}', skipping.")
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
        print(f"[experiments] No rows collected for experiment {name}")
        return None

    # Save per-experiment CSV
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXPERIMENTS_DIR / f"rl_dse_analysis_{name}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[experiments] Saved per-scenario metrics to {out_csv}")

    # Print aggregate summary
    agg = df.mean(numeric_only=True)
    print("\n[experiments] Aggregate metrics over all scenarios:")
    print(agg)

    return df

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    temperatures = load_temperatures()
    print("Temperature scenarios:", temperatures)

    # One-time rebuild for baseline DSE etc.
    run_cmd("make clean")
    run_cmd("make")

    all_dfs = []
    for exp in EXPERIMENTS:
        df = run_experiment(exp, temperatures)
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
