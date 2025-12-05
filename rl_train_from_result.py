import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


# Must match SAVE_DIR in model.py
SAVE_DIR = "/var/tmp/learning_thermostat"


class ThermostatPolicy(nn.Sequential):
    """
    Same architecture as self.model in model.py:

        nn.Sequential(
            nn.Linear(5, 64),
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

    We subclass nn.Sequential so that state_dict() has keys like "0.weight",
    "0.bias", ... exactly as in model.py, so load_state_dict() will work.
    """

    def __init__(self):
        super().__init__(
            nn.Linear(5, 64),
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


def build_dataset(
    df: pd.DataFrame,
    t_desired: float,
    ll: float,
    ul: float,
    cool_down: float,
    hyst: float,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    Build an offline dataset from result.csv.

    We use the baseline trajectory only:
        time, baseline_heater, baseline_T_bair, ...

    State s_t = [
        T_bair(t),
        T_desired - LL,
        T_desired + UL,
        1 if time_since_last_switch > C else -1,
        1 if time_since_last_switch > H else -1,
    ]

    Action a_t = 1 if baseline_heater else 0

    Reward r_t = - alpha * (T_bair - T_desired)^2 - beta * a_t
    (comfort vs energy trade-off)

    Returns are normalized discounted returns G_t.
    """

    times = df["time"].to_numpy(dtype=float)
    temps = df["baseline_T_bair"].to_numpy(dtype=float)
    heaters = df["baseline_heater"].to_numpy()
    actions = np.array([1.0 if h else 0.0 for h in heaters], dtype=float)

    n = len(df)
    states = np.zeros((n, 5), dtype=float)
    rewards = np.zeros(n, dtype=float)
    returns = np.zeros(n, dtype=float)

    if n == 0:
        raise ValueError("Empty dataframe, nothing to train on")

    last_switch_time = times[0]

    for i in range(n):
        t = times[i]
        a = actions[i]
        temp = temps[i]

        # update last_switch_time when the heater status changes
        if i > 0 and actions[i] != actions[i - 1]:
            last_switch_time = t

        time_since_switch = t - last_switch_time
        cd_flag = 1.0 if time_since_switch > cool_down else -1.0
        h_flag = 1.0 if time_since_switch > hyst else -1.0

        states[i, :] = [
            temp,
            t_desired - ll,
            t_desired + ul,
            cd_flag,
            h_flag,
        ]
        comfort_error = (temp - t_desired) ** 2
        energy = a
        rewards[i] = -alpha * comfort_error - beta * energy

    # compute discounted returns backwards
    G = 0.0
    for i in reversed(range(n)):
        G = rewards[i] + gamma * G
        returns[i] = G

    # normalize returns for stability
    returns_mean = returns.mean()
    returns_std = returns.std()
    if returns_std < 1e-8:
        returns_std = 1.0
    returns = (returns - returns_mean) / returns_std

    return states, actions, returns, rewards


def train_bc(
    states: np.ndarray,
    actions: np.ndarray,
    lr: float,
    epochs: int,
    device: str = "cpu",
    init_policy=None,
):
    """
    Behavioral cloning: supervised imitation of the baseline's heater decisions.

    Loss = BCE( pi(s_t), a_t )
    """
    device_t = torch.device(device)
    if init_policy is None:
        policy = ThermostatPolicy().to(device_t)
    else:
        policy = init_policy.to(device_t)

    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    bce = nn.BCELoss()

    states_t = torch.tensor(states, dtype=torch.float32, device=device_t)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device_t)

    for epoch in range(epochs):
        probs_on = policy(states_t).squeeze(-1)  # [N]
        probs_on = torch.clamp(probs_on, 1e-4, 1.0 - 1e-4)

        loss = bce(probs_on, actions_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            with torch.no_grad():
                pred_actions = (probs_on >= 0.5).float()
                acc = (pred_actions == actions_t).float().mean().item()
            print(
                f"[BC Epoch {epoch+1:4d}/{epochs}] "
                f"loss={loss.item():.4f}  "
                f"action_accuracy={acc:.3f}"
            )

    return policy


def train_rl(
    states: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    lr: float,
    epochs: int,
    device: str = "cpu",
    init_policy=None,
):
    """
    REINFORCE-style policy gradient on the offline dataset.

    We maximize E[ G_t * log pi(a_t | s_t) ].
    """
    device_t = torch.device(device)
    if init_policy is None:
        policy = ThermostatPolicy().to(device_t)
    else:
        policy = init_policy.to(device_t)

    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    states_t = torch.tensor(states, dtype=torch.float32, device=device_t)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device_t)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device_t)

    for epoch in range(epochs):
        probs_on = policy(states_t).squeeze(-1)  # [N]
        probs_on = torch.clamp(probs_on, 1e-4, 1.0 - 1e-4)

        dist = torch.distributions.Bernoulli(probs=probs_on)
        log_probs = dist.log_prob(actions_t)

        loss = -(log_probs * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            with torch.no_grad():
                pred_actions = (probs_on >= 0.5).float()
                acc = (pred_actions == actions_t).float().mean().item()
                avg_return = returns_t.mean().item()
            print(
                f"[RL Epoch {epoch+1:4d}/{epochs}] "
                f"loss={loss.item():.4f}  "
                f"action_accuracy={acc:.3f}  "
                f"avg_normalized_return={avg_return:.3f}"
            )

    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Offline BC + RL training for the thermostat policy using result.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="build/cmp/result.csv",
        help="Path to result.csv (must contain time, baseline_heater, baseline_T_bair columns)",
    )
    parser.add_argument(
        "--t_desired",
        type=float,
        default=25.5,
        help="Desired room temperature (should match the one used in simulation)",
    )
    parser.add_argument(
        "--ll",
        type=float,
        default=0.5,
        help="Lower comfort band (LL), such that lower bound = T_desired - LL",
    )
    parser.add_argument(
        "--ul",
        type=float,
        default=0.5,
        help="Upper comfort band (UL), such that upper bound = T_desired + UL",
    )
    parser.add_argument(
        "--cool_down",
        type=float,
        default=600.0,
        help="Cooldown time C (seconds) used in the state mask (time since last switch)",
    )
    parser.add_argument(
        "--hyst",
        type=float,
        default=1200.0,
        help="Hysteresis time H (seconds) used in the state mask (time since last switch)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Comfort weight in the reward: -alpha * (T - T_desired)^2",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Energy weight in the reward: -beta * heater_on",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for returns",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rl", "bc", "bc_then_rl"],
        default="rl",
        help=(
            "Training mode: "
            "'bc' (behavioral cloning only), "
            "'rl' (RL only, from scratch or init_from), "
            "'bc_then_rl' (pretrain with BC then fine-tune with RL)."
        ),
    )
    parser.add_argument(
        "--bc_epochs",
        type=int,
        default=1000,
        help="Number of epochs for behavioral cloning (if used)",
    )
    parser.add_argument(
        "--rl_epochs",
        type=int,
        default=1000,
        help="Number of epochs for RL fine-tuning (if used)",
    )
    parser.add_argument(
        "--bc_lr",
        type=float,
        default=1e-3,
        help="Learning rate for behavioral cloning",
    )
    parser.add_argument(
        "--rl_lr",
        type=float,
        default=1e-3,
        help="Learning rate for RL fine-tuning",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Optional path to an existing thermostat_nn_model.pt to initialize from",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    required_cols = {"time", "baseline_heater", "baseline_T_bair"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    states, actions, returns, rewards = build_dataset(
        df,
        t_desired=args.t_desired,
        ll=args.ll,
        ul=args.ul,
        cool_down=args.cool_down,
        hyst=args.hyst,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    print("Dataset built:")
    print(f"  steps       : {len(df)}")
    print(f"  reward mean : {rewards.mean():.3f}")
    print(f"  reward std  : {rewards.std():.3f}")

    device = args.device

    # Optional initialization from an existing model (e.g., online supervised)
    init_policy = None
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"init_from path does not exist: {args.init_from}")
        print(f"Initializing policy from {args.init_from}")
        init_policy = ThermostatPolicy()
        state_dict = torch.load(args.init_from, map_location="cpu")
        init_policy.load_state_dict(state_dict)

    final_policy = None

    if args.mode == "bc":
        print("=== Behavioral Cloning only ===")
        final_policy = train_bc(
            states=states,
            actions=actions,
            lr=args.bc_lr,
            epochs=args.bc_epochs,
            device=device,
            init_policy=init_policy,
        )

    elif args.mode == "rl":
        print("=== RL only (no BC pretraining) ===")
        final_policy = train_rl(
            states=states,
            actions=actions,
            returns=returns,
            lr=args.rl_lr,
            epochs=args.rl_epochs,
            device=device,
            init_policy=init_policy,
        )

    elif args.mode == "bc_then_rl":
        print("=== BC pretraining stage ===")
        bc_policy = train_bc(
            states=states,
            actions=actions,
            lr=args.bc_lr,
            epochs=args.bc_epochs,
            device=device,
            init_policy=init_policy,
        )
        print("=== RL fine-tuning stage ===")
        final_policy = train_rl(
            states=states,
            actions=actions,
            returns=returns,
            lr=args.rl_lr,
            epochs=args.rl_epochs,
            device=device,
            init_policy=bc_policy,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save final policy where model.py expects it
    final_policy.eval()
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "thermostat_nn_model.pt")
    torch.save(final_policy.state_dict(), save_path)
    print(f"Saved trained policy state_dict to: {save_path}")


if __name__ == "__main__":
    main()
