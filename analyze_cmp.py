import pandas as pd

T_DESIRED = 20.0  # o il valore che usi in simulazione

df = pd.read_csv("build/cmp/result.csv")

# Differenze tra baseline e RL
diff_mask = df["baseline_heater"] != df["ml_heater"]
n_diff = diff_mask.sum()
n_total = len(df)

print(f"Steps totali: {n_total}")
print(f"Steps in cui RL ≠ baseline: {n_diff} ({n_diff / n_total * 100:.2f}%)")

# Comfort: errore quadratico medio rispetto al setpoint
baseline_mse = ((df["baseline_T_bair"] - T_DESIRED) ** 2).mean()
ml_mse = ((df["ml_T_bair"] - T_DESIRED) ** 2).mean()

print(f"Baseline comfort MSE: {baseline_mse:.3f}")
print(f"RL comfort MSE      : {ml_mse:.3f}")

# Uso dell'heater come proxy di energia
baseline_on_ratio = df["baseline_heater"].mean()
ml_on_ratio = df["ml_heater"].mean()

print(f"Baseline heater ON ratio: {baseline_on_ratio:.3f}")
print(f"RL heater ON ratio      : {ml_on_ratio:.3f}")
