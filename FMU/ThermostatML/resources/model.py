"""
SIMPLIFIED VERSION - BASIC RL
Goal: Make RL work as fine-tuning of BC
UPDATE: support for methodologically correct BC vs RL comparison
"""
import os
import random
from typing import Optional, Dict, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fmi2 import Fmi2FMU, Fmi2Status
from SlidingWindowStepModel import SWSM


class PolicyNet(nn.Module):
    """Binary policy network with Sigmoid output - NO DROPOUT for control consistency."""
    def __init__(self, state_dim: int = 6):  # Now 6D: added dT/dt
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),  # Removed Dropout - critical for eval consistency
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(x), 1e-6, 1.0 - 1e-6)
    
    def freeze_base_layers(self):
        """Freeze all layers except the last Linear layer for fine-tuning."""
        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze ONLY the last Linear layer
        # New architecture (no Dropout): [0-1-2-3-4-5-6-7-8-9]
        # Layer 8 is Linear(64, 1), layer 9 is Sigmoid (no weights)
        self.net[8].weight.requires_grad = True
        self.net[8].bias.requires_grad = True
        
        print("[PolicyNet] Froze base layers, only last layer trainable")


class SimpleRL:
    """Ultra-simple policy gradient RL for fine-tuning WITH SCHEDULED BC REGULARIZATION."""
    def __init__(self, actor: PolicyNet):
        self.actor = actor
        
        # Moderate learning rate for last layer fine-tuning
        self.lr = 3e-6  # Conservative but allows learning
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # Short-term memory
        self.memory = deque(maxlen=20)
        self.batch_size = 10
        
        # BC regularization schedule: start strong, decay gradually
        self.beta_start = 10.0  # Strong initial constraint
        self.beta_end = 1.0     # Allow more RL freedom later
        self.beta_current = self.beta_start
        self.beta_decay = 0.998  # Slow decay
    
    def add_experience(self, state, action, reward, baseline_action, baseline_reward):
        """Store experience with baseline reward for advantage calculation."""
        self.memory.append((state, action, reward, baseline_action, baseline_reward))
    
    def update(self) -> Optional[Dict[str, float]]:
        """Policy gradient update with ADVANTAGE relative to BC baseline."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Get recent experiences
        batch = list(self.memory)[-self.batch_size:]
        
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        baseline_actions = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        baseline_rewards = torch.tensor([x[4] for x in batch], dtype=torch.float32)
        
        # ADVANTAGE: reward improvement over BC baseline
        advantages = rewards - baseline_rewards
        
        # Clip outliers
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient
        probs = self.actor(states).squeeze()
        
        # Log probability of taken action
        log_probs = torch.where(
            actions > 0.5,
            torch.log(probs + 1e-8),
            torch.log(1.0 - probs + 1e-8)
        )
        
        # Policy gradient loss with ADVANTAGE (not raw reward)
        pg_loss = -(log_probs * advantages).mean()
        
        # BC REGULARIZATION: stay close to baseline
        bc_loss = F.binary_cross_entropy(
            probs, 
            baseline_actions, 
            reduction='mean'
        )
        
        # Total loss with scheduled beta
        total_loss = pg_loss + self.beta_current * bc_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.optimizer.step()
        
        # Decay beta slowly
        self.beta_current = max(self.beta_end, self.beta_current * self.beta_decay)
        
        return {
            "loss": float(total_loss.item()),
            "pg_loss": float(pg_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "beta": float(self.beta_current),
            "avg_advantage": float(advantages.mean().item())
        }


def _normalize_policy_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Backward compatibility for old state_dicts."""
    if not sd:
        return sd
    if any(k.startswith("net.") for k in sd.keys()):
        return sd
    return {f"net.{k}": v for k, v in sd.items()}


class Model(Fmi2FMU):
    def __init__(self, reference_to_attr: Optional[dict] = None) -> None:
        super().__init__(reference_to_attr)
        
        # Inputs
        self.LL_in = -1.0
        self.UL_in = -1.0
        self.H_in = -1.0
        self.C_in = -1.0
        self.T_bair_in = -1.0
        self.heater_on_in = False
        
        # Outputs
        self.heater_on_out = False
        
        # Config
        self.T_desired = 25.5
        self.SAVE_DIR = os.getenv("THERMOSTAT_RUN_DIR", "/var/tmp/learning_thermostat")
        self.MODE = os.getenv("THERMOSTAT_MODE", "train")
        
        # Training phases
        self.BC_PHASE_END = 3000.0
        self.RL_PHASE_END = 7000.0   # Longer for convergence
        self.EVAL_PHASE_END = 10000.0
        
        # BC config
        self.bc_batch_size = 33
        
        # RL config - ULTRA CONSERVATIVE
        self.epsilon = 0.005         # Minimal exploration
        self.epsilon_decay = 0.999   # Very slow decay
        self.min_epsilon = 0.001
        
        # State
        self.phase = "bc"
        self.has_learnt = False
        self.bc_loss = 201.0
        
        self.last_heater_status = False
        self.commutation_time = 0.0
        self.last_time = float("-inf")
        self.last_temp = None
        
        # For BC balancing
        self.n_true = 0
        self.n_false = 0
        
        # Previous state for RL
        self.prev_state = None
        self.prev_action = None
        self.prev_time = None
        
        # Metrics
        self.total_switches = 0
        self.total_heating_time = 0.0
        self.emergency_count = 0
        self.rl_update_count = 0
        self.rl_update_every = 50  # Update every N steps
        
        # Paths
        self.POLICY_BC_PATH = os.path.join(self.SAVE_DIR, "policy_bc.pth")
        self.POLICY_RL_PATH = os.path.join(self.SAVE_DIR, "policy_rl.pth")
        self.POLICY_ACTIVE_PATH = os.path.join(self.SAVE_DIR, "policy_active.pth")
        
        # Initialize neural network
        self.model = PolicyNet(state_dim=6)
        self.model.eval()  # Start in eval mode for BC
        
        # BC trainer (SWSM)
        self.swsm = SWSM(
            self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=3e-5),
            window_size=self.bc_batch_size
        )
        
        # RL trainer (initialized later)
        self.simple_rl: Optional[SimpleRL] = None
        
        # EVAL mode: load RL policy if available
        if self.MODE == "eval":
            policy_path = self.POLICY_RL_PATH
            if os.path.exists(policy_path):
                try:
                    sd = torch.load(policy_path, map_location="cpu", weights_only=True)
                    sd = _normalize_policy_state_dict(sd)
                    self.model.load_state_dict(sd, strict=True)
                    self.model.eval()
                    self.phase = "eval"
                    print(f"[ThermostatML] EVAL MODE: Loading RL policy from {policy_path}")
                    print(f"[ThermostatML] Loaded RL policy successfully")
                    print(f"[ThermostatML] Starting evaluation with RL policy")
                except Exception as e:
                    print(f"[ThermostatML] ERROR loading RL policy: {e}")
                    raise
            else:
                raise FileNotFoundError(f"[ThermostatML] EVAL mode but no policy at {policy_path}")
        else:
            # TRAINING mode
            print("[ThermostatML] TRAINING MODE: Starting BC phase")
    
    def _build_features(self, time: float) -> torch.Tensor:
        """Build 6D state vector: [T, LL, UL, H, C, dT/dt]"""
        T = float(self.T_bair_in)
        LL = self.T_desired + float(self.LL_in)
        UL = self.T_desired + float(self.UL_in)
        H = float(self.H_in)
        C = float(self.C_in)
        
        # Temperature derivative (rate of change)
        if self.last_temp is not None and time > 0:
            dT_dt = T - self.last_temp  # °C per second (1s timestep)
        else:
            dT_dt = 0.0
        
        return torch.tensor([T, LL, UL, H, C, dT_dt], dtype=torch.float32)
    
    def _safe_action(self, time: float, x: torch.Tensor, action: int) -> Tuple[int, bool]:
        """
        HARD CONSTRAINT: enforce comfort bounds [LL, UL].
        Returns: (safe_action, was_overridden)
        """
        T = float(x[0].item())
        LL = float(x[1].item())
        UL = float(x[2].item())
        
        was_overridden = False
        
        # Hard constraint on LL and UL
        if T < LL:
            # Temperature too cold: MUST heat
            if action != 1:
                was_overridden = True
                self.emergency_count += 1
                if self.emergency_count % 50 == 1:  # Log occasionally
                    print(f"[SAFETY @ {time:.1f}s] Hard constraint: T={T:.1f}°C < LL={LL:.1f}°C → force heat")
            action = 1
        
        elif T > UL:
            # Temperature too hot: MUST NOT heat
            if action != 0:
                was_overridden = True
                self.emergency_count += 1
                if self.emergency_count % 50 == 1:  # Log occasionally
                    print(f"[SAFETY @ {time:.1f}s] Hard constraint: T={T:.1f}°C > UL={UL:.1f}°C → force cool")
            action = 0
        
        # Track switches
        if action != self.last_heater_status:
            self.total_switches += 1
            self.last_heater_status = action
        
        return action, was_overridden
    
    def _compute_reward(self, time: float, x: torch.Tensor, action: int, 
                       baseline_action: int, was_overridden: bool) -> float:
        """
        Continuous reward function with multiple components:
        1. Comfort: smooth reward inside zone, penalty outside
        2. Energy: light penalty for heating
        3. Anticipation: reward smart switching
        4. Override: penalty if safety intervened
        5. Extreme temps: catastrophic penalty
        """
        T = float(x[0].item())
        LL = float(x[1].item())
        UL = float(x[2].item())
        dT_dt = float(x[5].item())  # Temperature derivative
        
        # 1. COMFORT - Continuous reward
        if LL <= T <= UL:
            # Inside comfort zone: reward proximity to T_desired
            # Max reward=1.0 at T_desired, decreases smoothly toward boundaries
            distance_from_desired = abs(T - self.T_desired)
            comfort_band_width = (UL - LL) / 2.0
            comfort_reward = 1.0 - 0.5 * (distance_from_desired / comfort_band_width)
        else:
            # Outside comfort zone: strong penalty proportional to distance
            if T < LL:
                distance = LL - T
            else:
                distance = T - UL
            comfort_reward = -10.0 * distance  # Proportional penalty
        
        # 2. ENERGY - Light penalty for heating
        energy_penalty = -0.1 if action == 1 else 0.0
        
        # 3. ANTICIPATORY SWITCHING - Bonus for smart switching
        anticipation_reward = 0.0
        
        # Reward heating when cooling fast and temperature is below mid-point
        if dT_dt < -0.15 and action == 1 and T < self.T_desired:
            anticipation_reward = 0.3
        
        # Reward stopping when heating fast and temperature is above mid-point  
        elif dT_dt > 0.15 and action == 0 and T > self.T_desired:
            anticipation_reward = 0.3
        
        # Penalize late switching (already violated comfort)
        if T < LL and action == 0:
            anticipation_reward = -1.0  # Too cold but not heating
        elif T > UL and action == 1:
            anticipation_reward = -1.0  # Too hot but still heating
        
        # 4. OVERRIDE PENALTY - Policy tried something unsafe
        override_penalty = -5.0 if was_overridden else 0.0
        
        # 5. EXTREME TEMPS - Catastrophic (shouldn't happen with hard constraints)
        extreme_penalty = -50.0 if (T < LL - 1.0 or T > UL + 1.0) else 0.0
        
        total = comfort_reward + energy_penalty + anticipation_reward + override_penalty + extreme_penalty
        return float(np.clip(total, -100.0, 2.0))
    
    def ctrl_step(self, time: float):
        """Main control step."""
        if time < self.last_time + 1:
            return
        self.last_time = time
        
        # Track heating time
        if self.heater_on_out and self.phase in ["rl", "eval"]:
            self.total_heating_time += 1.0
        
        # Phase transitions
        if time >= self.RL_PHASE_END and self.phase == "rl":
            print(f"[ThermostatML] Entering EVAL phase @ {time:.1f}s")
            self.phase = "eval"
            self.model.eval()
            torch.save(self.model.state_dict(), self.POLICY_RL_PATH)
            print(f"[ThermostatML] Saved RL policy to {self.POLICY_RL_PATH}")
            print(f"[ThermostatML] Switches: {self.total_switches}, Heating time: {self.total_heating_time:.0f}s")
            print(f"[ThermostatML] RL updates performed: {self.rl_update_count}")
        
        elif time >= self.BC_PHASE_END and self.phase == "bc":
            print(f"[ThermostatML] Entering RL phase @ {time:.1f}s")
            self.phase = "rl"
            self.has_learnt = True
            self.model.eval()  # KEEP eval mode even in RL for consistency
            torch.save(self.model.state_dict(), self.POLICY_BC_PATH)
            print(f"[ThermostatML] Saved BC policy to {self.POLICY_BC_PATH}")
            
            # Initialize SIMPLE RL
            self.simple_rl = SimpleRL(self.model)
            
            # FREEZE base BC layers - only fine-tune last layer
            self.model.freeze_base_layers()
            
            # KEEP model.eval() for consistency (dropout removed, but good practice)
            # PyTorch allows backward even in eval mode
            print(f"[ThermostatML] Starting SIMPLE RL with epsilon={self.epsilon:.4f}")
            print(f"[ThermostatML] Strategy: ULTRA-CONSERVATIVE fine-tuning")
            print(f"[ThermostatML] - Beta schedule: {self.simple_rl.beta_start} → {self.simple_rl.beta_end}")
            print(f"[ThermostatML] - Updates every {self.rl_update_every} steps")
            print(f"[ThermostatML] - Mode: FROZEN BASE + EVAL MODE for consistency")
        
        x = self._build_features(time)
        
        # BC PHASE
        if self.phase == "bc":
            if (self.n_false != 0 and not self.heater_on_in and 
                ((self.n_false / (self.n_true + self.n_false)) > 0.5 or random.random() <= 0.1)):
                return
            
            if self.heater_on_in:
                self.n_true += 1
            else:
                self.n_false += 1
            
            target = torch.tensor([1.0 if self.heater_on_in else 0.0], dtype=torch.float32)
            _, loss = self.swsm.accumulate_and_step(x, training=True, target_tensor=target)
            if loss is not None:
                self.bc_loss = float(loss.item())
            
            self.heater_on_out = bool(self.heater_on_in)
            self.last_temp = float(self.T_bair_in)
            return
        
        # RL PHASE
        if self.phase == "rl":
            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    p = float(self.model(x.unsqueeze(0)).item())
                action = 1 if p >= 0.5 else 0
            
            # Get baseline action (what BC would do)
            baseline_action = int(self.heater_on_in)
            
            # Safety check on RL action
            action, was_overridden = self._safe_action(time, x, action)
            action = int(action)
            self.heater_on_out = bool(action)
            
            # Compute reward for RL action
            reward = self._compute_reward(time, x, action, baseline_action, was_overridden)
            
            # Compute reward for baseline action (what BC would have gotten)
            # Assume BC wouldn't be overridden (it's trained to be safe)
            baseline_reward = self._compute_reward(time, x, baseline_action, baseline_action, was_overridden=False)
            
            # Store experience WITH baseline reward for advantage calculation
            self.simple_rl.add_experience(x, action, reward, baseline_action, baseline_reward)
            
            # Update periodically
            if int(time) % self.rl_update_every == 0:
                stats = self.simple_rl.update()
                if stats:
                    self.rl_update_count += 1
                    if self.rl_update_count % 20 == 0:
                        print(f"[RL @ {time:.1f}s] loss={stats['loss']:.3f}, "
                              f"pg={stats['pg_loss']:.3f}, bc={stats['bc_loss']:.3f}, β={stats['beta']:.2f}, "
                              f"adv={stats['avg_advantage']:+.2f}, "
                              f"T={float(x[0].item()):.1f}°C, dT/dt={float(x[5].item()):+.2f}, "
                              f"r={reward:.1f}, switches={self.total_switches}")
            
            # Decay exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            self.prev_state = x
            self.prev_action = action
            self.prev_time = time
            self.last_temp = float(self.T_bair_in)
            return
        
        # EVAL PHASE
        if self.phase == "eval":
            with torch.no_grad():
                p = float(self.model(x.unsqueeze(0)).item())
            action = 1 if p >= 0.5 else 0
            
            action, was_overridden = self._safe_action(time, x, action)
            self.heater_on_out = bool(action)
            
            if was_overridden and int(time) % 500 == 0:
                print(f"[EVAL @ {time:.1f}s] Safety override triggered, T={float(x[0].item()):.1f}°C")
            
            return
    
    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(float(current_time))
        return Fmi2Status.ok
    
    def terminate(self) -> int:
        """Save final policies."""
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        
        try:
            torch.save(self.model.state_dict(), self.POLICY_ACTIVE_PATH)
            print(f"[ThermostatML] Saved active policy to {self.POLICY_ACTIVE_PATH}")
            
            if self.phase == "eval" and os.path.exists(self.POLICY_RL_PATH):
                print(f"[ThermostatML] RL policy available at {self.POLICY_RL_PATH}")
            elif self.phase == "bc" and os.path.exists(self.POLICY_BC_PATH):
                print(f"[ThermostatML] BC policy available at {self.POLICY_BC_PATH}")
            
            print(f"[ThermostatML SIMPLE] Training summary:")
            print(f"  - Final phase: {self.phase}")
            print(f"  - Total switches: {self.total_switches}")
            print(f"  - Total heating time: {self.total_heating_time:.0f}s")
            print(f"  - Emergency overrides: {self.emergency_count}")
            if self.phase == "eval":
                print(f"  - RL updates performed: {self.rl_update_count}")
        except Exception as e:
            print(f"[ThermostatML] ERROR saving: {e}")
        
        return Fmi2Status.ok