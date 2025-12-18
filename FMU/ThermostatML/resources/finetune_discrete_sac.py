#!/usr/bin/env python3
"""
finetune_discrete_sac.py

Discrete (binary) Soft Actor-Critic fine-tuning for the incubator thermostat.

Key design choices:
- Action space: {0, 1} (heater OFF/ON).
- Actor: Bernoulli policy pi(a=1|s)=p(s). This can be warm-started from your BC model
  that already outputs a probability via Sigmoid.
- Critics: twin Q networks output Q(s,0) and Q(s,1) (vector of size 2).
- Reward: combines comfort, energy, switch penalty, and an Rswitch penalty for late commutations.

CSV expectations:
- Needs a time column and enough signals to reconstruct the 5-D state:
  [T_bair, T_desired - LL, T_desired + UL, cooldown_flag, heatup_flag]
- It tries to auto-detect columns by name. If it fails, pass explicit --col-* arguments.

Typical use in DSE (called after each scenario run):
  python3 finetune_discrete_sac.py \
      --csv build/dse/23.5/stage2/outputs.csv \
      --policy-in /var/tmp/learning_thermostat/thermostat_policy.pt \
      --policy-out /var/tmp/learning_thermostat/thermostat_policy.pt \
      --sac-out /var/tmp/learning_thermostat/sac_binary.pt \
      --mode train --updates 400 --batch-size 256

Then your next DSE episode loads the updated policy weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Missing dependency: pandas. Install with `pip install pandas`.") from e

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit("Missing dependency: torch. Install PyTorch for your environment.") from e


# ---------------------------
# Utilities: column detection
# ---------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_column(cols: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    """
    Return first column name whose normalized form contains ANY normalized pattern.
    """
    ncols = [(_norm(c), c) for c in cols]
    npats = [_norm(p) for p in patterns]
    for nc, orig in ncols:
        if any(p in nc for p in npats):
            return orig
    return None


# ---------------------------
# Discrete SAC implementation
# ---------------------------

class BernoulliActor(nn.Module):
    """
    Binary policy pi(a=1|s)=p(s).
    Architecture mirrors your BC policy (5 -> ... -> 1 -> Sigmoid),
    so you can warm-start from the BC state_dict directly.
    """
    def __init__(self, state_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
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

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # returns p in (0,1)
        p = self.net(s)
        # numerical stability
        return torch.clamp(p, 1e-6, 1.0 - 1e-6)

    def probs_and_logprobs(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pi:     (B,2)  [P(a=0), P(a=1)]
          log_pi: (B,2)  [logP(a=0), logP(a=1)]
        """
        p1 = self.forward(s)  # (B,1)
        p0 = 1.0 - p1
        pi = torch.cat([p0, p1], dim=1)  # (B,2)
        log_pi = torch.log(pi)
        return pi, log_pi

    @torch.no_grad()
    def act(self, s_np: np.ndarray, evaluate: bool = False) -> int:
        """
        If evaluate=True: greedy (p>=0.5).
        Else: sample from Bernoulli(p).
        """
        s = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)
        p1 = self.forward(s).item()
        if evaluate:
            return 1 if p1 >= 0.5 else 0
        return 1 if np.random.rand() < p1 else 0


class TwinQ(nn.Module):
    """
    Two critics. Each outputs Q(s,0) and Q(s,1).
    """
    def __init__(self, state_dim: int = 5, hidden_dim: int = 256):
        super().__init__()

        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(s), self.q2(s)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    lr: float = 3e-4
    hidden_dim: int = 256
    auto_entropy: bool = False
    target_entropy: float = -0.5  # for binary, a modest negative target (tunable)


class DiscreteSAC:
    def __init__(self, state_dim: int, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.actor = BernoulliActor(state_dim).to(device)
        self.critic = TwinQ(state_dim, cfg.hidden_dim).to(device)
        self.critic_target = TwinQ(state_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        if cfg.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.log_alpha = None
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        if self.cfg.auto_entropy:
            return self.log_alpha.exp()
        return torch.tensor(self.cfg.alpha, device=self.device)

    def update(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        sp: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, float]:
        """
        s:    (B, state_dim)
        a:    (B, 1) int64 in {0,1}
        r:    (B, 1)
        sp:   (B, state_dim)
        done: (B, 1) float {0,1}
        """
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        sp = sp.to(self.device)
        done = done.to(self.device)

        # ------------------
        # Critic target
        # ------------------
        with torch.no_grad():
            pi_sp, logpi_sp = self.actor.probs_and_logprobs(sp)  # (B,2)
            q1_sp, q2_sp = self.critic_target(sp)
            qmin_sp = torch.min(q1_sp, q2_sp)  # (B,2)
            v_sp = (pi_sp * (qmin_sp - self.alpha * logpi_sp)).sum(dim=1, keepdim=True)  # (B,1)
            y = r + (1.0 - done) * self.cfg.gamma * v_sp

        q1_s, q2_s = self.critic(s)
        q1_sa = q1_s.gather(1, a)
        q2_sa = q2_s.gather(1, a)

        critic_loss = F.mse_loss(q1_sa, y) + F.mse_loss(q2_sa, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ------------------
        # Actor update
        # ------------------
        pi_s, logpi_s = self.actor.probs_and_logprobs(s)
        q1_s2, q2_s2 = self.critic(s)
        qmin_s = torch.min(q1_s2, q2_s2)

        actor_loss = (pi_s * (self.alpha * logpi_s - qmin_s)).sum(dim=1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ------------------
        # Alpha update (optional)
        # ------------------
        alpha_loss_val = 0.0
        if self.cfg.auto_entropy:
            # expected log pi = sum_a pi(a|s) log pi(a|s)
            expected_logpi = (pi_s * logpi_s).sum(dim=1).detach().mean()
            alpha_loss = -(self.log_alpha * (expected_logpi + self.cfg.target_entropy))
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        # ------------------
        # Soft target update
        # ------------------
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # Diagnostics
        ent = float((-(pi_s * logpi_s).sum(dim=1).mean()).item())
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss_val),
            "entropy": ent,
        }

    def load_policy_state_dict(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(sd)

    def save_policy_state_dict(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def save_full(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load_full(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_target.load_state_dict(ck["critic_target"])
        self.actor_opt.load_state_dict(ck["actor_opt"])
        self.critic_opt.load_state_dict(ck["critic_opt"])
        if self.cfg.auto_entropy and ck.get("log_alpha") is not None:
            self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))


# ---------------------------
# Data preparation & reward
# ---------------------------

def compute_rswitch_penalties(
    time: np.ndarray,
    tbair: np.ndarray,
    ll: np.ndarray,
    h: np.ndarray,
    action: np.ndarray,
    tol: float,
    w_switch: float,
) -> np.ndarray:
    """
    Rswitch penalty: -w_switch (indicator) if a commutation happens late.
    Two late cases:
      1) Turning ON late after temperature first goes below LL while heater is OFF.
      2) Turning OFF late after expected_off = t_on + H.

    The penalty is applied at the time index where the late commutation is observed.
    """
    n = len(time)
    rs = np.zeros(n, dtype=np.float32)

    # Case 1: late ON when T < LL
    expected_on_time: Optional[float] = None
    for i in range(n):
        if action[i] == 0 and expected_on_time is None and tbair[i] < ll[i]:
            expected_on_time = float(time[i])

        if expected_on_time is not None and action[i] == 1:
            delay = float(time[i]) - expected_on_time
            if delay > tol:
                rs[i] -= float(w_switch)
            expected_on_time = None

    # Case 2: late OFF after H seconds of heating
    expected_off_time: Optional[float] = None
    for i in range(n):
        if i == 0:
            prev = action[i]
        else:
            prev = action[i - 1]

        # Detect ON edge
        if prev == 0 and action[i] == 1:
            expected_off_time = float(time[i]) + float(h[i])

        # Detect OFF edge
        if expected_off_time is not None and prev == 1 and action[i] == 0:
            delay = float(time[i]) - expected_off_time
            if delay > tol:
                rs[i] -= float(w_switch)
            expected_off_time = None

    # If we never turned off by end and we are past expected_off_time => penalize at last step
    if expected_off_time is not None:
        if float(time[-1]) - expected_off_time > tol:
            rs[-1] -= float(w_switch)

    return rs


def build_states_from_csv(
    df: "pd.DataFrame",
    *,
    t_desired: float,
    col_time: str,
    col_tbair: str,
    col_ll: str,
    col_ul: str,
    col_h: str,
    col_c: str,
    col_action: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the 5-D state vector used in your FMU model:
      [T_bair, T_desired - LL, T_desired + UL, cooldown_flag, heatup_flag]
    where flags depend on (time - commutation_time) compared with C and H.
    """
    time = df[col_time].to_numpy(dtype=np.float32)
    tbair = df[col_tbair].to_numpy(dtype=np.float32)
    ll = df[col_ll].to_numpy(dtype=np.float32)
    ul = df[col_ul].to_numpy(dtype=np.float32)
    h = df[col_h].to_numpy(dtype=np.float32)
    c = df[col_c].to_numpy(dtype=np.float32)

    # action can be bool/float/int in CSV -> convert to {0,1}
    a_raw = df[col_action].to_numpy()
    if a_raw.dtype == np.bool_:
        action = a_raw.astype(np.int64)
    else:
        action = (a_raw.astype(np.float32) >= 0.5).astype(np.int64)

    # commutation time reconstructed from action edges
    comm_time = float(time[0])
    prev_a = int(action[0])

    s = np.zeros((len(df), 5), dtype=np.float32)
    for i in range(len(df)):
        if i > 0:
            if int(action[i]) != prev_a:
                comm_time = float(time[i])
                prev_a = int(action[i])

        cooldown_flag = 1.0 if (float(time[i]) - comm_time) > float(c[i]) else -1.0
        heatup_flag = 1.0 if (float(time[i]) - comm_time) > float(h[i]) else -1.0

        s[i, 0] = float(tbair[i])
        s[i, 1] = float(t_desired - ll[i])
        s[i, 2] = float(t_desired + ul[i])
        s[i, 3] = float(cooldown_flag)
        s[i, 4] = float(heatup_flag)

    return time, tbair, ll, ul, h, c, action, s


def compute_rewards(
    tbair: np.ndarray,
    action: np.ndarray,
    rswitch: np.ndarray,
    *,
    t_desired: float,
    w_comfort: float,
    w_energy: float,
    w_switch: float,
) -> np.ndarray:
    """
    Base reward (per time step):
      r = - w_comfort * (T - T_desired)^2
          - w_energy  * 1[heater_on]
          - w_switch  * 1[action changes]
          + rswitch   (already negative if late)

    Note: rswitch is applied at the timestep where the late commutation is observed.
    """
    n = len(tbair)
    r = np.zeros(n, dtype=np.float32)

    prev_a = int(action[0])
    for i in range(n):
        comfort = (float(tbair[i]) - float(t_desired)) ** 2
        energy = 1.0 if int(action[i]) == 1 else 0.0
        switch_evt = 1.0 if (i > 0 and int(action[i]) != prev_a) else 0.0

        r[i] = (
            - float(w_comfort) * float(comfort)
            - float(w_energy) * float(energy)
            - float(w_switch) * float(switch_evt)
            + float(rswitch[i])
        )

        prev_a = int(action[i])

    return r


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to outputs.csv (ideally stage2/outputs.csv).")

    # Optional explicit column mapping
    ap.add_argument("--col-time", default="", help="Time column name.")
    ap.add_argument("--col-tbair", default="", help="T_bair column name.")
    ap.add_argument("--col-ll", default="", help="LL column name.")
    ap.add_argument("--col-ul", default="", help="UL column name.")
    ap.add_argument("--col-h", default="", help="H column name.")
    ap.add_argument("--col-c", default="", help="C column name.")
    ap.add_argument("--col-action", default="", help="Executed action column (ML heater).")

    # Reward parameters
    ap.add_argument("--t-desired", type=float, default=25.5)
    ap.add_argument("--w-comfort", type=float, default=1.0)
    ap.add_argument("--w-energy", type=float, default=0.05)
    ap.add_argument("--w-switch", type=float, default=0.01)
    ap.add_argument("--w-switch-delay", type=float, default=0.05, help="w_switch for Rswitch delay penalty (indicator).")
    ap.add_argument("--switch-tol", type=float, default=0.0, help="Tolerance seconds; if 0, inferred from dt/2.")

    # SAC parameters
    ap.add_argument("--updates", type=int, default=400, help="Gradient updates to run.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--auto-entropy", action="store_true")
    ap.add_argument("--target-entropy", type=float, default=-0.5)

    # Persistence
    ap.add_argument("--policy-in", default="", help="Path to BC policy state_dict (.pt) to warm-start actor.")
    ap.add_argument("--policy-out", default="", help="Where to save updated policy state_dict (actor).")
    ap.add_argument("--sac-in", default="", help="Load full SAC checkpoint (optional).")
    ap.add_argument("--sac-out", default="", help="Save full SAC checkpoint (optional).")

    # Reporting
    ap.add_argument("--metrics-out", default="", help="Write a metrics.json next to scenario.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = list(df.columns)

    # Auto-detect columns if not provided
    col_time = args.col_time or find_column(cols, ["time", "t"])
    if not col_time:
        raise SystemExit("Could not detect time column. Pass --col-time.")

    col_tbair = args.col_tbair or find_column(cols, ["tbair", "t_bair", "plantinstance.tbair", "t_air", "airtemp"])
    if not col_tbair:
        # as fallback, sometimes T_bair might be logged as just "T_bair_out"
        col_tbair = find_column(cols, ["tbairout", "tbair_out"])
    if not col_tbair:
        raise SystemExit("Could not detect T_bair column. Pass --col-tbair.")

    col_ll = args.col_ll or find_column(cols, ["ll", "lowerlimit"])
    col_ul = args.col_ul or find_column(cols, ["ul", "upperlimit"])
    col_h = args.col_h or find_column(cols, ["h", "heatingduration"])
    col_c = args.col_c or find_column(cols, ["c", "waitingduration"])

    missing = [name for name, col in [("LL", col_ll), ("UL", col_ul), ("H", col_h), ("C", col_c)] if not col]
    if missing:
        raise SystemExit(
            f"Could not detect columns for {missing}. "
            f"Pass --col-ll/--col-ul/--col-h/--col-c explicitly."
        )

    # Action column: prefer ThermostatML output if present
    col_action = args.col_action or (
        find_column(cols, ["thermostatmlinstance.heater", "thermostatml", "mlheater", "ml.heater"])
        or find_column(cols, ["heater_on_out", "heateronout", "heater"])
    )
    if not col_action:
        raise SystemExit("Could not detect action column. Pass --col-action.")

    # Build states
    time, tbair, ll, ul, h, c, action, s = build_states_from_csv(
        df,
        t_desired=args.t_desired,
        col_time=col_time,
        col_tbair=col_tbair,
        col_ll=col_ll,
        col_ul=col_ul,
        col_h=col_h,
        col_c=col_c,
        col_action=col_action,
    )

    # Infer dt and tolerance
    if len(time) >= 2:
        dt = float(np.median(np.diff(time)))
    else:
        dt = 1.0
    tol = float(args.switch_tol) if args.switch_tol > 0 else 0.5 * dt

    # Rswitch penalty (late commutations)
    rswitch = compute_rswitch_penalties(
        time=time,
        tbair=tbair,
        ll=ll,
        h=h,
        action=action,
        tol=tol,
        w_switch=args.w_switch_delay,
    )

    # Per-step reward
    r = compute_rewards(
        tbair=tbair,
        action=action,
        rswitch=rswitch,
        t_desired=args.t_desired,
        w_comfort=args.w_comfort,
        w_energy=args.w_energy,
        w_switch=args.w_switch,
    )

    # Build transitions (s_t, a_t, r_t, s_{t+1}, done)
    # done only at final transition
    s_t = s[:-1]
    s_tp1 = s[1:]
    a_t = action[:-1].astype(np.int64).reshape(-1, 1)
    r_t = r[:-1].astype(np.float32).reshape(-1, 1)
    done = np.zeros((len(s_t), 1), dtype=np.float32)
    done[-1, 0] = 1.0

    # Torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SACConfig(
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        auto_entropy=args.auto_entropy,
        target_entropy=args.target_entropy,
    )
    agent = DiscreteSAC(state_dim=s_t.shape[1], cfg=cfg, device=device)

    # Warm-start actor from BC policy
    if args.policy_in and os.path.exists(args.policy_in):
        agent.load_policy_state_dict(args.policy_in)

    # Optionally resume full SAC
    if args.sac_in and os.path.exists(args.sac_in):
        agent.load_full(args.sac_in)

    # Training loop: sample minibatches from the episode buffer
    S = torch.tensor(s_t, dtype=torch.float32)
    A = torch.tensor(a_t, dtype=torch.int64)
    R = torch.tensor(r_t, dtype=torch.float32)
    SP = torch.tensor(s_tp1, dtype=torch.float32)
    D = torch.tensor(done, dtype=torch.float32)

    n = S.shape[0]
    if n < args.batch_size:
        raise SystemExit(f"Not enough transitions ({n}) for batch_size={args.batch_size}.")

    stats_acc = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0, "alpha_loss": 0.0, "entropy": 0.0}
    for u in range(args.updates):
        idx = torch.randint(0, n, (args.batch_size,))
        st = S[idx]
        at = A[idx]
        rt = R[idx]
        spt = SP[idx]
        dt_ = D[idx]
        st_out = agent.update(st, at, rt, spt, dt_)
        for k in stats_acc:
            stats_acc[k] += float(st_out[k])

    for k in stats_acc:
        stats_acc[k] /= float(args.updates)

    # Episode-level KPIs (simple)
    ep_return = float(r_t.sum())
    comfort_rmse = float(math.sqrt(np.mean((tbair - args.t_desired) ** 2)))
    energy_frac = float(np.mean(action.astype(np.float32)))
    switch_count = int(np.sum(action[1:] != action[:-1]))
    late_switch_count = int(np.sum(rswitch < 0.0))

    metrics = {
        "csv": args.csv,
        "columns": {
            "time": col_time,
            "tbair": col_tbair,
            "ll": col_ll,
            "ul": col_ul,
            "h": col_h,
            "c": col_c,
            "action": col_action,
        },
        "dt": dt,
        "switch_tol": tol,
        "reward_weights": {
            "w_comfort": args.w_comfort,
            "w_energy": args.w_energy,
            "w_switch": args.w_switch,
            "w_switch_delay": args.w_switch_delay,
        },
        "episode": {
            "return": ep_return,
            "comfort_rmse": comfort_rmse,
            "energy_frac_on": energy_frac,
            "switch_count": switch_count,
            "late_switch_count": late_switch_count,
        },
        "sac": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "avg_critic_loss": stats_acc["critic_loss"],
            "avg_actor_loss": stats_acc["actor_loss"],
            "alpha": stats_acc["alpha"],
            "avg_alpha_loss": stats_acc["alpha_loss"],
            "entropy": stats_acc["entropy"],
            "auto_entropy": bool(args.auto_entropy),
        },
    }

    # Save updated policy (actor only) for ThermostatML to load next run
    if args.policy_out:
        agent.save_policy_state_dict(args.policy_out)

    # Save full SAC checkpoint (optional)
    if args.sac_out:
        agent.save_full(args.sac_out)

    # Write metrics file
    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Print minimal summary
    print(json.dumps(metrics["episode"], indent=2))
    print(json.dumps(metrics["sac"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
