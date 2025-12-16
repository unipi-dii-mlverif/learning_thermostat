import os
import random

import torch
from torch import nn

from fmi2 import Fmi2FMU, Fmi2Status
from SlidingWindowStepModel import SWSM
from sac_agent import SACAgent


class Model(Fmi2FMU):
    """
    Thermostat FMU model with:
    - Online supervised learning via SlidingWindow (behavior cloning)
    - Optional offline RL loading for the BC model (USE_OFFLINE_RL)
    - Online RL (SAC) for multi-scenario training and evaluation (USE_SAC / RL_ENABLED + MODE)
    """

    def __init__(self, reference_to_attr=None) -> None:
        super().__init__(reference_to_attr)

        # ===== FMU inputs / outputs =====
        # Inputs
        self.LL_in = -1.0
        self.UL_in = -1.0
        self.H_in = -1.0
        self.C_in = -1.0
        self.T_bair_in = -1.0
        self.heater_on_in = False  # supervisor / baseline command (for BC)

        # Output
        self.heater_on_out = False  # final command

        # ===== High-level configuration =====
        self.T_desired = 25.5

        # Persistence / model I/O
        self.SAVE_TO_DISK = True
        self.SAVE_DIR = "/var/tmp/learning_thermostat"
        self.model_load_path = ""   # can be overridden by mm*.json
        self.model_save_path = ""   # mainly for SAC checkpoints

        # BC / offline-RL configuration (vecchio flag)
        self.USE_OFFLINE_RL = False

        # RL configuration (per DSE RL online)
        # MODE:
        #   "BC"        -> SlidingWindow only (default)
        #   "RL_TRAIN"  -> SAC online training
        #   "EVAL"      -> SAC policy-only evaluation
        self.MODE = "BC"

        # Questi due possono essere settati via mm_dse_rl / mm_dse_eval:
        self.USE_SAC = False         # se true, chiede di usare SAC
        self.RL_ENABLED = False      # flag alternativo (alcune config potrebbero usare questo nome)
        self.RL_MODE_ACTIVE = False  # flag interno: "SAC effettivamente attivo"

        # Supervised model state
        self.has_learnt = False
        self.loss = 201.0
        self.threshold = 0.002
        self.model_training = True

        # Time / hysteresis bookkeeping
        self.last_heater_status = False
        self.commutation_time = 0.0
        self.last_time = float("-inf")

        # Simple balancing counters (BC path)
        self.n_true = 0
        self.n_false = 0

        # ===== Supervised NN used by Sliding Window (BC) =====
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
            nn.Sigmoid(),  # outputs prob(heater_on = 1)
        )

        # Sliding-window trainer (supervised)
        self.swsm = SWSM(self.model, 33)

        # ===== RL (SAC) structures =====
        self.sac_agent: SACAgent | None = None
        self.sac_checkpoint_path: str = ""

        # RL transition bookkeeping
        self.rl_prev_state = None       # list[float]
        self.rl_prev_action = None      # numpy array shape [1]
        self.rl_prev_heater_on = False
        self.rl_step_counter = 0

        # RL hyperparameters
        self.RL_BATCH_SIZE = 256
        self.RL_UPDATES_PER_STEP = 1
        self.RL_UPDATE_INTERVAL = 1  # call update every N env steps

        # Reward weights
        self.reward_alpha = 1.0           # comfort
        self.reward_beta = 0.05           # energy
        self.reward_lambda_switch = 0.01  # commutation penalty

    # ------------------------------------------------------------------
    # Initialisation / mode selection
    # ------------------------------------------------------------------

    def _init_sac_agent(self):
        if self.sac_agent is None:
            # state_dim = 5 (same as BC input), action_dim = 1 (heater in [0,1])
            self.sac_agent = SACAgent(state_dim=5, action_dim=1)

    def exit_initialization_mode(self):
        """
        Configure the controller at the end of FMU initialization.

        Priority:
        1) If USE_SAC or RL_ENABLED and MODE in {"RL_TRAIN", "EVAL"} -> Online RL with SAC.
        2) Else if USE_OFFLINE_RL -> load pre-trained BC model from disk.
        3) Else -> original behaviour: online SlidingWindow only, optional load if model_load_path is set.
        """

        # --------- 1) SAC path (online RL) -----------
        external_rl_flag = bool(self.USE_SAC) or bool(self.RL_ENABLED)
        self.RL_MODE_ACTIVE = external_rl_flag and (self.MODE in ("RL_TRAIN", "EVAL"))

        if self.RL_MODE_ACTIVE:
            print(f"[Model] Initializing SAC in MODE={self.MODE}, USE_SAC={self.USE_SAC}, RL_ENABLED={self.RL_ENABLED}")

            self._init_sac_agent()

            # Decide checkpoint path for SAC
            if self.model_load_path and self.model_load_path.strip():
                self.sac_checkpoint_path = self.model_load_path
            else:
                self.sac_checkpoint_path = os.path.join(self.SAVE_DIR, "thermostat_sac_model.pt")

            # Try to load existing SAC checkpoint
            if os.path.exists(self.sac_checkpoint_path):
                try:
                    print(f"[Model] Loading SAC agent from {self.sac_checkpoint_path}...")
                    self.sac_agent.load(self.sac_checkpoint_path)
                    print(f"[Model] SAC agent loaded successfully from {self.sac_checkpoint_path}")
                except Exception as e:
                    print(f"[Model] Error loading SAC agent from {self.sac_checkpoint_path}: {e}")
                    print("[Model] Starting SAC from scratch.")
            else:
                print(f"[Model] No SAC checkpoint found at {self.sac_checkpoint_path}, starting from scratch.")

            # In SAC mode we treat the policy as "already learnt" from the FMU point of view
            self.has_learnt = True
            self.model_training = False
            self.loss = 0.0

            return Fmi2Status.ok

        # --------- 2) Offline RL / BC model loading -----------
        if self.USE_OFFLINE_RL:
            # Attempt to auto-resolve model path if not provided
            if not (self.model_load_path and self.model_load_path.strip()):
                default_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
                if os.path.exists(default_path):
                    print(f"[Model] No model_load_path provided, but found pretrained weights at {default_path}")
                    self.model_load_path = default_path

            if self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading BC model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.has_learnt = True
                    self.model_training = False
                    self.loss = 0.0
                    print(f"[Model] BC model loaded successfully from {self.model_load_path}")
                except Exception as e:
                    print(f"[Model] Error loading BC model from {self.model_load_path}: {e}")
                    print("[Model] Falling back to online supervised training.")

            return Fmi2Status.ok

        # --------- 3) Original behaviour: SlidingWindow only -----------
        if self.model_load_path and self.model_load_path.strip():
            try:
                print(f"[Model] Loading BC model from {self.model_load_path}...")
                state_dict = torch.load(self.model_load_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_learnt = True
                self.model_training = False
                self.loss = 0.0
                print(f"[Model] BC model loaded successfully from {self.model_load_path}")
            except Exception as e:
                print(f"[Model] Error loading BC model from {self.model_load_path}: {e}")
                print("[Model] Continuing with online supervised training.")

        return Fmi2Status.ok

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _build_state_vector(self, time: float):
        """Build the 5-D state vector used by both BC and RL."""
        return [
            float(self.T_bair_in),
            float(self.T_desired - self.LL_in),
            float(self.T_desired + self.UL_in),
            1.0 if (time - self.commutation_time) > self.C_in else -1.0,
            1.0 if (time - self.commutation_time) > self.H_in else -1.0,
        ]

    def _compute_reward(self, current_temp: float, heater_on: bool, prev_heater_on: bool) -> float:
        """Compute scalar reward for SAC (comfort + energy + commutations)."""
        comfort_error = (current_temp - self.T_desired) ** 2
        energy = 1.0 if heater_on else 0.0
        switch_penalty = 1.0 if heater_on != prev_heater_on else 0.0

        r = (
            - self.reward_alpha * comfort_error
            - self.reward_beta * energy
            - self.reward_lambda_switch * switch_penalty
        )
        return float(r)

    # ------------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------------

    def ctrl_step(self, time: float):
        # Simple anti-oversampling (keep original behaviour)
        if time < self.last_time + 1:
            return
        self.last_time = time

        if self.RL_MODE_ACTIVE:
            self._ctrl_step_rl(time)
        else:
            self._ctrl_step_bc(time)

    # ----- Behavioural Cloning / SlidingWindow path -----

    def _ctrl_step_bc(self, time: float):
        # Determine current heater status (baseline vs learned)
        heater_status = self.heater_on_in if not self.has_learnt else self.heater_on_out
        if self.last_heater_status != heater_status:
            self.commutation_time = time
        self.last_heater_status = heater_status

        # Update simple counts (kept for compatibility)
        if self.heater_on_in:
            self.n_true += 1
        else:
            self.n_false += 1

        # Build state and target for BC
        state_vec = torch.tensor(self._build_state_vector(time), dtype=torch.float32)
        target = torch.tensor([1.0 if self.heater_on_in else 0.0], dtype=torch.float32)

        out, loss2 = self.swsm.accumulate_and_step(state_vec, self.model_training, target)
        if loss2 is not None:
            self.loss = loss2

        # Decide output: prima che il modello sia "done", segui il baseline
        if out is None or not self.has_learnt:
            self.heater_on_out = bool(self.heater_on_in)
        else:
            self.heater_on_out = bool(out[0] >= 0.5)

        # Keep loss as scalar
        if self.has_learnt and self.loss is None:
            self.loss = torch.tensor([-self.threshold])

        if self.loss is not None and not isinstance(self.loss, float):
            self.loss = float(self.loss.item())

        # Simple stopping criterion for BC training
        if ((time >= 3000 and self.loss <= self.threshold) or time >= 9000) and not self.has_learnt:
            print("=========================> BC DONE @", time)
            self.has_learnt = True
            self.model_training = False

    # ----- RL (SAC) path -----

    def _ctrl_step_rl(self, time: float):
        if self.sac_agent is None:
            # Safety: should not happen if exit_initialization_mode configured correctly
            self._init_sac_agent()

        # 1) Build current state
        current_state = self._build_state_vector(time)

        # 2) If we have a previous transition, compute reward and push to replay buffer
        if self.rl_prev_state is not None and self.rl_prev_action is not None:
            prev_heater_on = bool(self.rl_prev_action[0] > 0.5)

            reward = self._compute_reward(
                current_temp=float(self.T_bair_in),
                heater_on=prev_heater_on,
                prev_heater_on=self.rl_prev_heater_on,
            )
            done = 0.0  # episodic termination handled at FMU reset

            self.sac_agent.replay_buffer.push(
                self.rl_prev_state,
                [float(self.rl_prev_action[0])],
                reward,
                current_state,
                done,
            )

            # SAC updates (only in training mode)
            if self.MODE == "RL_TRAIN":
                self.rl_step_counter += 1
                if self.rl_step_counter % self.RL_UPDATE_INTERVAL == 0:
                    self.sac_agent.update_parameters(
                        batch_size=self.RL_BATCH_SIZE,
                        updates=self.RL_UPDATES_PER_STEP,
                    )

        # 3) Select new action with SAC policy
        evaluate = (self.MODE == "EVAL")
        action = self.sac_agent.select_action(current_state, evaluate=evaluate)
        # action is numpy array of shape [1]
        heater_on = bool(action[0] > 0.5)

        # Update output and bookkeeping
        if self.last_heater_status != heater_on:
            self.commutation_time = time
        self.last_heater_status = heater_on

        self.heater_on_out = heater_on

        # Store for next transition
        self.rl_prev_state = current_state
        self.rl_prev_action = action
        self.rl_prev_heater_on = heater_on

    # ------------------------------------------------------------------
    # FMI entry points
    # ------------------------------------------------------------------

    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(current_time)
        return Fmi2Status.ok

    def terminate(self) -> int:
        # 1) Save SAC agent if we are in RL mode and saving is enabled
        if self.RL_MODE_ACTIVE and self.sac_agent is not None and self.SAVE_TO_DISK:
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            path = (self.model_save_path or self.sac_checkpoint_path) or os.path.join(
                self.SAVE_DIR, "thermostat_sac_model.pt"
            )
            print(f"[Model] Saving SAC agent to {path}...")
            try:
                self.sac_agent.save(path)
                print(f"[Model] SAC agent saved to {path}")
            except Exception as e:
                print(f"[Model] Error saving SAC agent to {path}: {e}")

        # 2) Save supervised BC model only if:
        #    - we are NOT in SAC mode
        #    - SAVE_TO_DISK is True
        #    - and model_load_path was not explicitly set (to avoid overwriting)
        if (not self.RL_MODE_ACTIVE) and self.SAVE_TO_DISK and not (
            self.model_load_path and self.model_load_path.strip()
        ):
            os.makedirs(self.SAVE_DIR, exist_ok=True)

            model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
            print(f"Saving BC model to {model_path}...")
            torch.save(self.model.state_dict(), model_path)
            print(f"BC model saved to {model_path}")

            full_model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_full_model.pt")
            torch.save(self.model, full_model_path)
            print(f"Full BC model saved to {full_model_path}")

        return Fmi2Status.ok
