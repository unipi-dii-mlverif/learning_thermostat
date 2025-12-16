import os
from typing import Optional

import torch
from torch import nn

from fmi2 import Fmi2FMU, Fmi2Status
from online_bc_rl_agent import OnlineBCRLAgent


class Model(Fmi2FMU):
    """
    Thermostat ML controller with a single shared policy network trained via:
    - Behaviour Cloning (BC) using the classical controller as teacher
    - Reinforcement Learning (RL) using a scalar reward

    The same policy is used across all DSE runs. At the end of each episode
    (one Maestro simulation), BC and RL updates are applied and the weights
    are saved to disk so that the next episode continues training from there.
    """

    def __init__(self, reference_to_attr: Optional[dict] = None) -> None:
        super().__init__(reference_to_attr)

        # =======================
        # FMU inputs / outputs
        # =======================
        # Inputs
        self.LL_in = -1.0
        self.UL_in = -1.0
        self.H_in = -1.0
        self.C_in = -1.0
        self.T_bair_in = -1.0
        self.heater_on_in = False  # teacher / baseline action

        # Outputs
        self.heater_on_out = False  # final commanded action

        # =======================
        # Configuration
        # =======================
        self.T_desired = 25.5
        # Reward weights (can be overridden via FMU parameters / DSE)
        self.reward_alpha = 1.0   # comfort weight
        self.reward_beta = 0.05   # energy weight
        self.reward_lambda = 0.01 # commutation weight

        # Persistence / model I/O
        self.SAVE_TO_DISK = True
        self.SAVE_DIR = "/var/tmp/learning_thermostat"
        self.model_load_path = ""
        self.model_save_path = ""

        # Legacy flag: if True and a model_load_path is given, we only load a
        # pre-trained model and disable online updates (pure inference).
        self.USE_OFFLINE_RL = False

        # High-level mode:
        #   "BC_ONLY" : only behaviour cloning (no RL); used for classic training.
        #   "BC_RL"   : joint BC + RL training per episode (for DSE RL runs).
        #   "EVAL"    : pure inference with frozen policy.
        self.MODE = "BC_ONLY"

        # Training bookkeeping
        self.has_learnt = False
        self.loss = 201.0  # exposed scalar; here we use BC loss as proxy
        self.threshold = 0.002
        self.model_training = True

        # Time / hysteresis bookkeeping
        self.last_heater_status = False
        self.commutation_time = 0.0
        self.last_time = float("-inf")

        # =======================
        # Shared policy network
        # =======================
        # Same architecture as in the original project
        self.model = nn.Sequential(
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
            nn.Sigmoid(),  # output prob(heater_on = 1)
        )

        # =======================
        # Joint BC + RL agent
        # =======================
        self.bc_rl_agent: Optional[OnlineBCRLAgent] = None

        # Hyper-parameters (can be tuned / exposed if needed)
        self.RL_GAMMA = 0.99
        self.BC_COEF = 1.0
        self.LR = 3e-4

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _ensure_agent(self, use_rl: bool) -> None:
        if self.bc_rl_agent is None:
            self.bc_rl_agent = OnlineBCRLAgent(
                policy=self.model,
                use_rl=use_rl,
                gamma=self.RL_GAMMA,
                bc_coef=self.BC_COEF,
                lr=self.LR,
                device="cpu",
            )

    def _build_state_vector(self, time: float):
        """
        Build the 5-D state vector used by the policy.
        This mirrors the features used in the original sliding-window model:
        [T_bair, T_desired - LL, T_desired + UL, cooldown_flag, heatup_flag]
        """
        cooldown_flag = 1.0 if (time - self.commutation_time) > self.C_in else -1.0
        heatup_flag = 1.0 if (time - self.commutation_time) > self.H_in else -1.0

        return [
            float(self.T_bair_in),
            float(self.T_desired - self.LL_in),
            float(self.T_desired + self.UL_in),
            cooldown_flag,
            heatup_flag,
        ]

    def _compute_reward(self, current_temp: float, heater_on: bool, prev_heater_on: bool) -> float:
        """
        Simple scalar reward:
        - penalizes squared temperature error (comfort),
        - penalizes energy when heater is ON,
        - penalizes commutations.
        The relative importance of these terms is controlled by:
        - self.reward_alpha
        - self.reward_beta
        - self.reward_lambda
        """
        comfort_error = (current_temp - self.T_desired) ** 2
        energy = 1.0 if heater_on else 0.0
        switch_penalty = 1.0 if heater_on != prev_heater_on else 0.0

        r = (
            - self.reward_alpha * comfort_error
            - self.reward_beta * energy
            - self.reward_lambda * switch_penalty
        )
        return float(r)


    # ------------------------------------------------------------------
    # FMI lifecycle hooks
    # ------------------------------------------------------------------

    def exit_initialization_mode(self):
        """
        Configure the controller at the end of FMU initialization.

        Behaviour:
        - If USE_OFFLINE_RL is True and model_load_path is set:
            -> load a pre-trained policy and disable further updates.
        - Else:
            -> create an OnlineBCRLAgent; if a model_load_path exists, load weights.
        """

        # Decide whether RL is active or not based on MODE
        rl_active = (self.MODE == "BC_RL")

        if self.USE_OFFLINE_RL:
            # Pure inference from a pre-trained model
            if self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading offline model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.has_learnt = True
                    self.model_training = False
                    self.loss = 0.0
                    print(f"[Model] Offline model loaded successfully from {self.model_load_path}")
                except Exception as e:
                    print(f"[Model] Error loading offline model from {self.model_load_path}: {e}")
                    print("[Model] Falling back to online BC/RL training.")
            return Fmi2Status.ok

        # Online BC / BC+RL path
        self._ensure_agent(use_rl=rl_active)
        # >>> AGGIUNGI QUESTO BLOCCO <<<
        # Per i run RL (BC_RL) e per la sola evaluation (EVAL),
        # vogliamo che ThermostatML sia considerato "pronto" fin da subito,
        # così mm2.json può eseguire il modelSwap immediatamente.
        if self.MODE in ("BC_RL", "EVAL"):
            self.has_learnt = True
            self.model_training = False
        # <<< FINE BLOCCO AGGIUNTO >>>
        # Handle model loading if a path is provided
        effective_load_path = None
        if self.model_load_path and self.model_load_path.strip():
            effective_load_path = self.model_load_path
        else:
            # Default location
            default_path = os.path.join(self.SAVE_DIR, "thermostat_policy.pt")
            if os.path.exists(default_path):
                effective_load_path = default_path

        if effective_load_path:
            try:
                print(f"[Model] Loading policy from {effective_load_path}...")
                state_dict = torch.load(effective_load_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_learnt = True
                self.model_training = False
                self.loss = 0.0
                print(f"[Model] Policy loaded successfully from {effective_load_path}")
            except Exception as e:
                print(f"[Model] Error loading policy from {effective_load_path}: {e}")
                print("[Model] Starting from random initialization.")

        return Fmi2Status.ok

    # ------------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------------

    def ctrl_step(self, time: float):
        """
        Single control step at simulator time `time`.

        - In BC_ONLY mode:
            * The policy acts, but only BC loss is used at episode end.
        - In BC_RL mode:
            * The policy acts and both BC + RL losses are used at episode end.
        - In EVAL mode:
            * The policy acts with frozen weights (no updates, no storage).
        """

        # Simple anti-oversampling: skip if calls are too close in time
        if time < self.last_time + 1:
            return
        self.last_time = time

        # Build state and teacher action
        state = self._build_state_vector(time)
        teacher_action = 1.0 if self.heater_on_in else 0.0
        prev_heater_on = bool(self.heater_on_out)

        if self.MODE == "EVAL":
            # Pure inference: policy decides, no learning
            _, p_on = self.model_forward(state)
            self.heater_on_out = bool(p_on >= 0.5)

            # Update commutation time bookkeeping
            if prev_heater_on != self.heater_on_out:
                self.commutation_time = time
            self.last_heater_status = self.heater_on_out
            return

        # Ensure agent is created (for BC_ONLY or BC_RL)
        rl_active = (self.MODE == "BC_RL")
        self._ensure_agent(use_rl=rl_active)

        # Policy action
        policy_action, _ = self.bc_rl_agent.select_policy_action(state)
        executed_action = policy_action

        # Apply action
        self.heater_on_out = bool(executed_action)

        # Update commutation bookkeeping
        if prev_heater_on != self.heater_on_out:
            self.commutation_time = time
        self.last_heater_status = self.heater_on_out

        # Reward for RL / logging
        reward = self._compute_reward(
            current_temp=float(self.T_bair_in),
            heater_on=self.heater_on_out,
            prev_heater_on=prev_heater_on,
        )

        # Store transition for BC/RL update at episode end
        self.bc_rl_agent.observe(
            state=state,
            teacher_action=teacher_action,
            executed_action=executed_action,
            reward=reward,
        )

        # 'loss' will be updated at terminate() from the BC loss

    def model_forward(self, state):
        """Utility for EVAL mode: forward pass without agent machinery."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_on = self.model(s).item()
        return (1.0 if p_on >= 0.5 else 0.0), p_on

    # ------------------------------------------------------------------
    # FMI do_step / terminate
    # ------------------------------------------------------------------

    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(current_time)
        return Fmi2Status.ok

    def terminate(self) -> int:
        """
        At the end of each Maestro run (episode), perform BC/RL updates
        via the agent and persist the policy to disk so that the next
        DSE episode starts from the updated controller.
        """
        # Perform learning step if agent exists and we are not in pure EVAL mode
        if self.bc_rl_agent is not None and self.MODE in ("BC_ONLY", "BC_RL"):
            stats = self.bc_rl_agent.end_episode_and_update(self.MODE)
            if stats is not None:
                # Expose BC loss as 'loss' for compatibility
                self.loss = stats["bc_loss"]
                self.has_learnt = True

        # Save the policy if requested
        if self.SAVE_TO_DISK and not self.USE_OFFLINE_RL:
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            # Decide where to save: explicit model_save_path, else model_load_path, else default
            if self.model_save_path and self.model_save_path.strip():
                save_path = self.model_save_path
            elif self.model_load_path and self.model_load_path.strip():
                save_path = self.model_load_path
            else:
                save_path = os.path.join(self.SAVE_DIR, "thermostat_policy.pt")

            print(f"[Model] Saving policy to {save_path}...")
            torch.save(self.model.state_dict(), save_path)
            print(f"[Model] Policy saved to {save_path}")

        return Fmi2Status.ok
