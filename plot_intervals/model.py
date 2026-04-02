import os
import torch
from torch import nn

from .config import SAVE_DIR, OFFSET, ALPHA, T_desired, UL, LL, C_flag, H_flag


def steady_state_integral_error(t):
    """Leaky-integrator steady-state: e/(1-α) where e = T - (T_desired + UL - OFFSET)."""
    return (t - (T_desired + UL - OFFSET)) / (1.0 - ALPHA)


def make_state_full(t, dT, c_flag, h_flag):
    """Build the 7-element state vector with explicit commutation flags."""
    return torch.Tensor([
        t - (T_desired + (UL - LL) / 2.0),
        t - (T_desired - LL),
        t - (T_desired + UL),
        c_flag,
        h_flag,
        dT,
        steady_state_integral_error(t),
    ])


def make_state(t, dT):
    """Build the 7-element state vector for a single (T, dT) point."""
    return make_state_full(t, dT, C_flag, H_flag)


def create_model():
    return nn.Sequential(
        nn.Linear(7, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def load_model():
    model = create_model()
    model_path = os.path.join(SAVE_DIR, "thermostat_nn_model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model
