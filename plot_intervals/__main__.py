import sys
import torch

from .config import T_bair, T_derivative
from .model import load_model, make_state
from .plots import (
    plot_temperature_intervals,
    plot_2d_region,
    plot_2d_temp_vs_tsc,
    plot_2d_tsc_vs_deriv,
)

DISPATCH = {
    '0': plot_temperature_intervals,
    '1': plot_2d_region,
    '2': plot_2d_temp_vs_tsc,
    '3': plot_2d_tsc_vs_deriv,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in DISPATCH:
        print("Usage: python -m plot_intervals <plot>")
        print("  0  — output intervals vs temperature")
        print("  1  — 2D region: temperature vs T_derivative")
        print("  2  — 2D region: temperature vs time-since-commutation")
        print("  3  — 2D region: time-since-commutation vs T_derivative")
        sys.exit(1)

    model = load_model()

    # Quick sanity-check inference
    with torch.no_grad():
        out = model(make_state(T_bair, T_derivative))
        print(f"\nSanity check — raw output: {out[0].item():.4f}, "
              f"heater ON: {(out[0] >= 0.5).item()}")

    DISPATCH[sys.argv[1]](model)


if __name__ == '__main__':
    main()
