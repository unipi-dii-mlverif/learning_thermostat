import pickle
import SlidingWindowStepModel
from fmi2 import Fmi2FMU, Fmi2Status
import torch
from torch import nn
import torch.nn.functional as F
import statistics
from SlidingWindowStepModel import *
import random
import os
import math
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple


class SimpleRL:
    """Simple policy gradient RL for fine-tuning with timing constraints."""
    def __init__(self, policy: nn.Module, lr: float = 1e-5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.memory = deque(maxlen=100)  # Store recent experiences
        self.gamma = 0.95  # Discount factor
        self.batch_size = 20
        
    def add_experience(self, state, action, reward, log_prob):
        """Store experience."""
        self.memory.append((state, action, reward, log_prob))
    
    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self) -> Optional[Dict[str, float]]:
        """Policy gradient update."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Get batch
        batch = list(self.memory)[-self.batch_size:]
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        rewards = [x[2] for x in batch]
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Recompute log probabilities for current policy
        self.policy.train()
        probs = self.policy(states).squeeze()
        
        # Log probability of taken action
        log_probs = torch.where(
            actions > 0.5,
            torch.log(probs + 1e-8),
            torch.log(1.0 - probs + 1e-8)
        )
        
        # Policy gradient loss
        loss = -(log_probs * returns).mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.policy.eval()
        
        return {
            "loss": float(loss.item()),
            "avg_return": float(returns.mean().item()),
            "avg_reward": float(np.mean(rewards))
        }


class Model(Fmi2FMU):
    def __init__(self, reference_to_attr=None) -> None:
        super().__init__(reference_to_attr)

        # FMU's inputs and outputs
        self.LL_in = -1.0
        self.UL_in = -1.0
        self.H_in = -1.0
        self.C_in = -1.0
        self.T_bair_in = -1.0
        self.heater_on_in = False # Used to learn
        self.heater_on_out = False # output
        self.has_learnt = False
        self.loss = 201.0
        self.model_load_path = ""
        self.T_desired = 25.5
        self.SAVE_TO_DISK = True
        self.SAVE_DIR = "/var/tmp/learning_thermostat"
        self.USE_OFFLINE_RL = False
        self.START_IN_EVAL_MODE = False
        
        self.last_heater_status = False
        self.commutation_time = 0
        self.last_time = float("-inf")
        self.threshold = 0.002 #0.001 
        self.model_training = True
        
        # Temperature derivative tracking
        self.prev_T = None
        self.prev_T_time = None
        self.T_derivative = 0.0
        
        # RL phase tracking
        self.phase = "bc"  # "bc" -> "rl" -> "eval"
        self.BC_PHASE_END = 3_000.0
        self.RL_PHASE_END = 12_000.0
        self.rl_agent: Optional[SimpleRL] = None
        self.rl_update_count = 0
        self.rl_update_every = 50  # Update every N steps
        self.epsilon = 0.25  # Exploration rate for RL
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Previous state for RL
        self.prev_state = None
        self.prev_action = None
        
        # Epsilon-greedy action persistence
        self.epsilon_action = None  # Store the epsilon-greedy action
        self.epsilon_action_time = float("-inf")  # When the epsilon action was taken
        self.epsilon_action_duration = 10.0  # Keep epsilon action for 8 seconds
        
        # Metrics
        self.total_switches = 0
        self.total_violations = 0  # Timing constraint violations
        self.model = nn.Sequential(
            nn.Linear(6, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            #nn.Softmax(dim=-1)
            nn.Sigmoid(),
        )
        self.n_true = 0
        self.n_false = 0

        self.swsm = SWSM(self.model, 33)
    
    def compute_reward(self, time: float, action: int, state: torch.Tensor) -> float:
        """Compute reward for RL with strong overshoot prevention and anticipatory behavior."""
        T = float(state[0].item())
        LL = float(state[1].item())
        UL = float(state[2].item())
        dT_dt = float(state[5].item())  # Temperature derivative
        time_since_comm = time - self.commutation_time
        
        reward = 0.0
        
        # 1. COMFORT REWARD/PENALTY: Stay within [LL, UL] with ASYMMETRIC penalties
        if 1.1*LL <= T <= 0.9*UL:
            # Gaussian-like reward peaking at center of comfort zone
            center = (LL + UL) / 2.0
            width = (1.1*UL - 0.9*LL) / 2.0
            # Gaussian: exp(-((T - center)^2) / (2 * sigma^2))
            # We want peak of 0.5, so we scale by 0.5
            # sigma = width/2 gives reasonable spread
            sigma = width / 2.0
            reward += 0.5 * math.exp(-((T - center) ** 2) / (2 * sigma ** 2))
            # Inside comfort zone - good!
            #if T > self.T_desired:
            #    reward -= 1.0 - 1.0*(UL - T)/self.UL_in
            #else:
            #    reward += 0.5*(T - LL)/self.LL_in
        #else:
        #    # Outside comfort zone - STRONG penalties, especially for overshooting UL
        #    if T < LL:
        #        distance = LL - T
        #        reward -= 8.0 * distance  # Moderate penalty for undershoot
        #    else:
        #        distance = T - UL
        #        # MUCH STRONGER penalty for exceeding upper limit (overshoot is worse)
        #        reward -= 20.0 * distance  # Very strong penalty
        #        # Extra penalty if still rising
        #        if dT_dt > 0:
        #            reward -= 10.0 * dT_dt
        
        # 2. PREDICTIVE/ANTICIPATORY REWARDS: Encourage turning off heater BEFORE overshooting
        # Predict temperature in next 30-60 seconds based on current derivative
        prediction_horizon = 45.0  # seconds
        predicted_T = T + dT_dt * prediction_horizon
        
        #if action == 1:  # Currently heating
        #    # If predicted temperature will exceed UL, strongly discourage continuing to heat
        #    if predicted_T > UL:
        #        margin = predicted_T - UL
        #        reward -= 5.0 * margin
        #    # If temperature is close to UL and rising, discourage heating
        #    elif T > UL - 1.5 and dT_dt > 0.02:
        #        reward -= 4.0
        #else:  # Currently not heating (action == 0)
        #    # Reward for staying off when temperature would overshoot
        #    if T > UL - 2.0 and dT_dt > 0:
        #        reward += 1.0
        #    # If predicted temperature will drop below LL, discourage staying off
        #    if predicted_T < LL:
        #        margin = LL - predicted_T
        #        reward -= 2.0 * margin
        
        # 3. DERIVATIVE-BASED PENALTIES: Penalize dangerous trends
        if T > UL and dT_dt > 0.01:  # Above limit and still rising - very bad!
            reward -= 100.0 * dT_dt
        elif T > 0.961*UL and dT_dt > 0.02:  # Rising too fast toward upper bound
            reward -= 72 * dT_dt
        
        # 4. TIMING CONSTRAINT REWARD
        # Only reward respecting timing, no penalty for switching too early
        switched = (action != self.last_heater_status)
        
        #if switched:
        #    if action == 1:  # Turning on
        #        min_time = self.C_in  # Should wait at least C_in after turning off
        #        if time_since_comm >= min_time:
        #            reward += 0.5  # Respected timing constraint
        #    else:  # Turning off
        #        min_time = self.H_in  # Should wait at least H_in after turning on
        #        if time_since_comm >= min_time:
        #            reward += 0.5  # Respected timing constraint
        #
        ## Penalty for NOT switching when temperature is out of bounds (switching too late)
        #if not switched:
        #    if T < LL and action == 0:
        #        # Temperature too cold but not heating
        #        reward -= 2.0
        #    elif T > UL and action == 1:
        #        # Temperature too hot but still heating
        #        reward -= 2.0
        #
        ## 3. ENERGY PENALTY (minor)
        #if action == 1:
        #    reward -= 0.05  # Small penalty for heating
        #
        ## 4. SWITCHING PENALTY (discourage too frequent switches)
        #if switched:
        #    reward -= 0.2
        
        #return float(np.clip(reward, -20.0, 2.0))
        return reward
    
    def exit_initialization_mode(self):
        """
        Decide se usare un modello pre-addestrato (offline RL)
        o il training online SlidingWindow come nel progetto originale.
        """
        
        # Skip directly to eval mode if requested
        if self.START_IN_EVAL_MODE:
            print("[Model] START_IN_EVAL_MODE=True, skipping to eval phase")
            self.phase = "eval"
            self.has_learnt = True
            self.model_training = False
            self.model.eval()
            
            # Try to load model if path provided
            if self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    print(f"[Model] Model loaded successfully")
                except Exception as e:
                    print(f"[Model] Warning: Could not load model: {e}")
            else:
                print("[Model] Warning: No model_load_path provided, using untrained model")
            
            return Fmi2Status.ok

        if self.USE_OFFLINE_RL:
            # === Modalità OFFLINE RL / BC+RL ===
            # 1) Se nessun path esplicito, prova a usare il default
            if not (self.model_load_path and self.model_load_path.strip()):
                default_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
                if os.path.exists(default_path):
                    print(f"[Model] No model_load_path provided, "
                          f"but found pretrained weights at {default_path}")
                    self.model_load_path = default_path

            # 2) Se abbiamo un path, proviamo a caricare
            if self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.has_learnt = True
                    self.model_training = False   # disabilita SlidingWindow
                    self.loss = 0.0
                    print(f"[Model] Model loaded successfully from {self.model_load_path}")
                except Exception as e:
                    print(f"[Model] Error loading model from {self.model_load_path}: {e}")
                    print("[Model] Falling back to online supervised training.")
                    # in caso di errore, ci si comporta come nel ramo sotto

        else:
            # === Modalità ORIGINALE: solo SlidingWindow online ===
            # Qui manteniamo il comportamento "vecchio":
            # carica un modello solo se model_load_path è stato esplicitamente settato
            if True: #self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    #self.model.eval()
                    #self.has_learnt = True
                    #self.model_training = False
                    #self.loss = 0.0
                    print(f"[Model] Model loaded successfully from {self.model_load_path}")
                except Exception as e:
                    print(f"[Model] Error loading model from {self.model_load_path}: {e}")
                    print("[Model] Continuing with online supervised training.")

        return Fmi2Status.ok

    def ctrl_step(self, time):
        if time < self.last_time + 1:
            return
        
        self.last_time = time
        
        # === PHASE TRANSITIONS ===
        if time >= self.RL_PHASE_END and self.phase == "rl":
            print(f"[RL->EVAL] Switching to eval phase @ {time:.1f}s")
            print(f"[RL STATS] Updates: {self.rl_update_count}, Switches: {self.total_switches}, Violations: {self.total_violations}")
            self.phase = "eval"
            self.model.eval()
            self.has_learnt = True
            # Save RL-tuned model
            if self.SAVE_TO_DISK:
                os.makedirs(self.SAVE_DIR, exist_ok=True)
                rl_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model_rl.pt")
                torch.save(self.model.state_dict(), rl_path)
                print(f"[RL] Saved RL-tuned model to {rl_path}")
            return
        
        if time >= self.BC_PHASE_END and self.phase == "bc":
            if self.loss <= self.threshold or time >= self.BC_PHASE_END + 100:
                print(f"[BC->RL] Switching to RL phase @ {time:.1f}s (loss={self.loss:.4f})")
                self.phase = "rl"
                self.has_learnt = True
                self.model_training = False
                self.model.eval()
                
                # Initialize RL agent
                self.rl_agent = SimpleRL(self.model, lr=1e-5)
                print(f"[RL] Initialized RL agent with epsilon={self.epsilon:.3f}")
                
                # Save BC model
                if self.SAVE_TO_DISK:
                    os.makedirs(self.SAVE_DIR, exist_ok=True)
                    bc_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model_bc.pt")
                    torch.save(self.model.state_dict(), bc_path)
                    print(f"[BC] Saved BC model to {bc_path}")
        
        # Calculate temperature derivative
        if self.prev_T is not None and self.prev_T_time is not None:
            dt = time - self.prev_T_time
            if dt > 0:
                self.T_derivative = (self.T_bair_in - self.prev_T) / dt
        
        self.prev_T = self.T_bair_in
        self.prev_T_time = time
        
        # Build state representation
        state = torch.Tensor([
            self.T_bair_in,
            self.T_desired - self.LL_in,
            self.T_desired + self.UL_in,
            math.tanh((time - self.commutation_time - self.C_in) / 2.0),
            math.tanh((time - self.commutation_time - self.H_in) / 2.0),
            self.T_derivative
        ])
        
        # === BC PHASE (Supervised Learning) ===
        if self.phase == "bc":
            heater_status = self.heater_on_in
            if self.last_heater_status != heater_status:
                self.commutation_time = time
            
            # Poor men's balancing
            if self.n_false != 0 and not self.heater_on_in and ((self.n_false / (self.n_true + self.n_false) > .5) or (random.random() <= 0.1)):
                return
            
            if self.heater_on_in:
                self.n_true += 1
            else:
                self.n_false += 1
            
            target = torch.Tensor([1.0 if self.heater_on_in else 0.0])
            out, loss2 = self.swsm.accumulate_and_step(state, self.model_training, target)
            if loss2 is not None:
                self.loss = loss2.item() if hasattr(loss2, 'item') else loss2
            
            if out is None:
                self.heater_on_out = True
            else:
                self.heater_on_out = out[0] >= 0.5
            
            self.last_heater_status = self.heater_on_out
            return
        
        # === RL PHASE (Policy Gradient Fine-tuning) ===
        if self.phase == "rl":
            # Epsilon-greedy action selection with persistence
            time_since_epsilon = time - self.epsilon_action_time
            
            if self.epsilon_action is not None and time_since_epsilon < self.epsilon_action_duration:
                # Use stored epsilon-greedy action (keep for 8 seconds)
                action = self.epsilon_action
            elif random.random() < self.epsilon:
                # New epsilon-greedy exploration
                action = random.randint(0, 1)
                self.epsilon_action = action
                self.epsilon_action_time = time
            else:
                # Policy-based action (greedy)
                with torch.no_grad():
                    prob = float(self.model(state.unsqueeze(0)).item())
                action = 1 if prob >= 0.5 else 0
                # Reset epsilon action tracking
                self.epsilon_action = None
            
            # Compute reward for previous step if we have one
            if self.prev_state is not None and self.prev_action is not None and self.rl_agent is not None:
                reward = self.compute_reward(time - 1, self.prev_action, self.prev_state)
                # Compute log prob for storage (not used in current simple implementation but kept for completeness)
                with torch.no_grad():
                    prob = float(self.model(self.prev_state.unsqueeze(0)).item())
                    log_prob = np.log(prob + 1e-8) if self.prev_action == 1 else np.log(1 - prob + 1e-8)
                
                self.rl_agent.add_experience(self.prev_state, self.prev_action, reward, log_prob)
            
            # Update policy periodically
            if int(time) % self.rl_update_every == 0 and self.rl_agent is not None:
                stats = self.rl_agent.update()
                if stats:
                    self.rl_update_count += 1
                    if self.rl_update_count % 10 == 0:
                        print(f"[RL @ {time:.1f}s] loss={stats['loss']:.3f}, "
                              f"avg_ret={stats['avg_return']:.2f}, avg_rew={stats['avg_reward']:.2f}, "
                              f"eps={self.epsilon:.3f}, T={self.T_bair_in:.1f}°C")
            
            # Decay exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Execute action
            self.heater_on_out = bool(action)
            
            # Track switches
            if self.heater_on_out != self.last_heater_status:
                self.total_switches += 1
                self.commutation_time = time
            
            # Store for next step
            self.prev_state = state
            self.prev_action = action
            self.last_heater_status = self.heater_on_out
            return
        
        # === EVAL PHASE (Pure Inference) ===
        if self.phase == "eval":
            with torch.no_grad():
                prob = float(self.model(state.unsqueeze(0)).item())
            action = 1 if prob >= 0.5 else 0
            self.heater_on_out = bool(action)
            
            if self.heater_on_out != self.last_heater_status:
                self.commutation_time = time
            
            self.last_heater_status = self.heater_on_out
            return

    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(current_time)

        return Fmi2Status.ok
    
    def terminate(self) -> int:
        # Save to disk only if we didn't load from disk
        if self.SAVE_TO_DISK:# and not (self.model_load_path and self.model_load_path.strip()):        
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            
            # Save the model state dict based on phase
            if self.phase == "rl" or self.phase == "eval":
                model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model_rl.pt")
                print(f"[RL/EVAL] Saving RL-tuned model to {model_path}...")
            else:
                model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
                print(f"[BC] Saving BC model to {model_path}...")
            
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Also save as default name for future loading
            default_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
            #if model_path != default_path:
            torch.save(self.model.state_dict(), default_path)
            print(f"Model also saved to {default_path} (default load path)")
            
            # Also save the entire model
            full_model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_full_model.pt")
            torch.save(self.model, full_model_path)
            print(f"Full model saved to {full_model_path}")
            
            # Print summary
            print(f"\n[TRAINING SUMMARY]")
            print(f"  Final phase: {self.phase}")
            print(f"  Total switches: {self.total_switches}")
            if self.phase in ["rl", "eval"]:
                print(f"  RL updates: {self.rl_update_count}")
                print(f"  Timing violations: {self.total_violations}")
            
        return Fmi2Status.ok
