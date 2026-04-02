import math

# ── Output directory ──────────────────────────────────────────────────────────

SAVE_DIR = "/var/tmp/learning_thermostat"

# ── Fixed parameters shared by all plots ─────────────────────────────────────

OFFSET = 0.6
ALPHA = 0.9

T_bair = 32.5
LL = 5.0
UL = 0.0
T_desired = 35
time_since_comm = 5
C_in = 30.0
H_in = 20.0
T_derivative = 0.05

C_flag = math.tanh((time_since_comm - C_in) / 2.0)
H_flag = math.tanh((time_since_comm - H_in) / 2.0)
