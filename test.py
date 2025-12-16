import pandas as pd

temp = "25.0"

df_base = pd.read_csv(f"build/dse/{temp}/outputs.csv")
df_rl   = pd.read_csv(f"build/rl_dse_eval/{temp}/outputs.csv")

T_col        = "{Plant}.PlantInstance.T_bair_out"
H_base_col   = "{Controller}.ControllerInstance.heater_on_out"
H_rl_col     = "{Plant}.PlantInstance.heater_on_in"

T_desired = 25.0

def summary(df, name):
    temp_err = (df[T_col] - T_desired).abs().mean()
    heater_on_ratio = df[H_rl_col].mean()  # azione applicata al Plant
    print(f"{name}: mean |T-T_desired| = {temp_err:.3f}, heater_on_ratio = {heater_on_ratio:.3f}")

summary(df_base, "BASELINE/DSE (BC only)")
summary(df_rl,   "RL_EVAL (BC+RL policy)")

# confronto tra azioni baseline e RL nella run RL_EVAL
h_teacher = df_rl[H_base_col]  # comando del Controller (teacher)
h_rl      = df_rl[H_rl_col]    # azione RL applicata al Plant
diff_ratio = (h_teacher != h_rl).mean()
print(f"Fraction of timesteps where RL action != BASELINE teacher: {diff_ratio:.3f}")
