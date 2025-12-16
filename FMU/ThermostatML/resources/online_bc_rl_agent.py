import torch
from torch import nn, optim
from typing import List, Dict, Optional


class OnlineBCRLAgent:
    """
    Joint Behaviour Cloning + RL agent operating on a shared policy network.

    The policy is assumed to output a probability p(on|s) in [0,1].
    Actions are Bernoulli(p).
    """

    def __init__(
        self,
        policy: nn.Module,
        use_rl: bool = True,
        gamma: float = 0.99,
        bc_coef: float = 1.0,
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.use_rl = use_rl
        self.gamma = gamma
        self.bc_coef = bc_coef
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.bce = nn.BCELoss()

        self.reset_episode()

    # ---------------- Episode storage ----------------

    def reset_episode(self) -> None:
        self.states: List[list[float]] = []
        self.teacher_actions: List[float] = []
        self.executed_actions: List[float] = []
        self.rewards: List[float] = []

    def select_policy_action(self, state) -> tuple[float, float]:
        """
        Given a state (array-like of floats), return:
        - discrete action in {0.0, 1.0}
        - probability p_on in [0,1]
        """
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            p_on = self.policy(s).item()
        p_on_clamped = max(1e-4, min(1.0 - 1e-4, p_on))
        action = 1.0 if p_on_clamped >= 0.5 else 0.0
        return action, p_on_clamped

    def observe(self, state, teacher_action: float, executed_action: float, reward: float) -> None:
        self.states.append(list(state))
        self.teacher_actions.append(float(teacher_action))
        self.executed_actions.append(float(executed_action))
        self.rewards.append(float(reward))

    # ---------------- Training ----------------

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns G_t with standard backward recursion."""
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        G = 0.0
        for t in reversed(range(T)):
            G = float(rewards[t]) + self.gamma * G
            returns[t] = G
        # Normalize for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def end_episode_and_update(self, mode: str) -> Optional[Dict[str, float]]:
        """
        Called once per episode (e.g., from FMU.terminate()).

        Performs:
        - Always: a BC step (if there is data).
        - If use_rl and mode indicates RL is active ("BC_RL"): an RL step afterwards.
          This approximates: first BC training, then RL fine-tuning within the same episode.
        """
        if not self.states:
            return None

        states = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        teacher = torch.tensor(self.teacher_actions, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.executed_actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        # Clear buffer for next episode
        self.reset_episode()

        # Forward pass for probabilities
        probs = self.policy(states).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)

        # ---- Behaviour Cloning loss ----
        bc_loss = self.bce(probs, teacher)
        total_loss = bc_loss * self.bc_coef

        # First optimizer step: BC
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        rl_loss_value = 0.0

        # ---- RL loss (REINFORCE) ----
        if self.use_rl and mode == "BC_RL":
            # Recompute probabilities after BC step
            probs = self.policy(states).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
            returns = self._compute_returns(rewards)
            dist = torch.distributions.Bernoulli(probs=probs)
            log_probs = dist.log_prob(actions)
            rl_loss = -(log_probs * returns).mean()

            self.optimizer.zero_grad()
            rl_loss.backward()
            self.optimizer.step()

            rl_loss_value = float(rl_loss.item())

        return {
            "bc_loss": float(bc_loss.item()),
            "rl_loss": rl_loss_value,
        }
