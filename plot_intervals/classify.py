import math

from torchverif.interval_tensor.v2 import IntervalTensor
from torchverif.net_interval.v2 import bounds_from_v2_predictions

from .config import C_in, H_in, T_bair
from .model import make_state, make_state_full, make_state_with_tdes

# Module-level model reference for worker processes (populated via fork).
_pool_model = None


def _classify(interval):
    """Run interval through the model and return 0/1/2."""
    o = _pool_model(interval)
    bounds = bounds_from_v2_predictions(o)
    out_lb = bounds[0][1]
    out_ub = bounds[1][1]
    if out_lb >= 0.5:
        return 2
    if out_ub >= 0.5:
        return 1
    return 0


def _classify_cell(args):
    """Worker: classify a single (T, dT) cell."""
    j, i, t_lo, t_hi, dt_lo, dt_hi = args
    return j, i, _classify(IntervalTensor(
        make_state(t_lo, dt_lo),
        make_state(t_hi, dt_hi),
    ))


def _classify_cell_temp_tsc(args):
    """Worker: classify a single (T, time-since-commutation) cell."""
    j, i, t_lo, t_hi, tsc_lo, tsc_hi, dt_lo, dt_hi = args
    c_lb = math.tanh((tsc_lo - C_in) / 2.0)
    c_ub = math.tanh((tsc_hi - C_in) / 2.0)
    h_lb = math.tanh((tsc_lo - H_in) / 2.0)
    h_ub = math.tanh((tsc_hi - H_in) / 2.0)
    return j, i, _classify(IntervalTensor(
        make_state_full(t_lo, dt_lo, c_lb, h_lb),
        make_state_full(t_hi, dt_hi, c_ub, h_ub),
    ))


def _classify_cell_tsc_deriv(args):
    """Worker: classify a single (time-since-commutation, dT) cell."""
    j, i, tsc_lo, tsc_hi, dt_lo, dt_hi = args
    c_lb = math.tanh((tsc_lo - C_in) / 2.0)
    c_ub = math.tanh((tsc_hi - C_in) / 2.0)
    h_lb = math.tanh((tsc_lo - H_in) / 2.0)
    h_ub = math.tanh((tsc_hi - H_in) / 2.0)
    return j, i, _classify(IntervalTensor(
        make_state_full(T_bair, dt_lo, c_lb, h_lb),
        make_state_full(T_bair, dt_hi, c_ub, h_ub),
    ))


def _classify_cell_t_vs_tdes(args):
    """Worker: classify a single (T, T_desired) cell."""
    j, i, t_lo, t_hi, tdes_lo, tdes_hi = args
    return j, i, _classify(IntervalTensor(
        make_state_with_tdes(t_lo, 0.0, tdes_lo),
        make_state_with_tdes(t_hi, 0.0, tdes_hi),
    ))
