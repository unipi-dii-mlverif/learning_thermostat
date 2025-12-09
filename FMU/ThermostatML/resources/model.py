import pickle
import SlidingWindowStepModel
from fmi2 import Fmi2FMU, Fmi2Status
import torch
from torch import nn
import statistics
from SlidingWindowStepModel import *
import random
import os

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
        
        self.last_heater_status = False
        self.commutation_time = 0
        self.last_time = float("-inf")
        self.threshold = 0.002 #0.001 
        self.model_training = True
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
            #nn.Softmax(dim=-1)
            nn.Sigmoid(),
        )
        self.n_true = 0
        self.n_false = 0

        self.swsm = SWSM(self.model, 33)
    def exit_initialization_mode(self):
        """
        Decide se usare un modello pre-addestrato (offline RL)
        o il training online SlidingWindow come nel progetto originale.
        """

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
            if self.model_load_path and self.model_load_path.strip():
                try:
                    print(f"[Model] Loading model from {self.model_load_path}...")
                    state_dict = torch.load(self.model_load_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.has_learnt = True
                    self.model_training = False
                    self.loss = 0.0
                    print(f"[Model] Model loaded successfully from {self.model_load_path}")
                except Exception as e:
                    print(f"[Model] Error loading model from {self.model_load_path}: {e}")
                    print("[Model] Continuing with online supervised training.")

        return Fmi2Status.ok

    def ctrl_step(self, time):
        if time < self.last_time + 1: #or True:
            return
        
        self.last_time = time

        #if not self.has_learnt:
        #    print(self.loss)

        heater_status = self.heater_on_in if not self.has_learnt else self.heater_on_out
        if self.last_heater_status != heater_status:
            #print("commuto @ ", time)
            self.commutation_time = time

        # Poor men's balancing
        if not self.has_learnt and self.n_false != 0 and not self.heater_on_in and ((self.n_false / (self.n_true + self.n_false) > .5) or (random.random() <= 0.1)):
            #print("skipping", self.n_false, self.n_true)
            return
        
        if self.heater_on_in:
            self.n_true += 1
        else:
            self.n_false += 1
        
        input = torch.Tensor([
            self.T_bair_in,
            self.T_desired-self.LL_in,
            self.T_desired+self.UL_in,
            #1 if self.H_in > 0 else -1,
            1 if time - self.commutation_time > self.C_in else -1,
            1 if time - self.commutation_time > self.H_in else -1
            #self.T_bair_in,
            #self.T_desired - self.LL_in,
            #time - self.commutation_time - self.C_in,
            #time - self.commutation_time - (self.C_in if heater_status else self.H_in) if heater_status and self.H_in > 0 else 0
            #self.T_bair_in - (self.T_desired - self.LL_in),
            #self.T_bair_in - (self.T_desired + self.UL_in)
        ])
        target = torch.Tensor([1.0 if self.heater_on_in else 0.0]) #, 1.0 if not self.heater_on_in else 0.0])
        #if self.heater_on_in:
        #    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", input, target)
        out, loss2 = self.swsm.accumulate_and_step(input, self.model_training, target)
        if loss2 is not None:
            self.loss = loss2
        
        #if loss2 is not None and abs(loss2 - self.loss) < 0.1:
        #    self.n_false = 0
        #    self.n_true = 0
        
        # Start outputing heating command when the model is trained and swapped
        if out is None:
            self.heater_on_out = True
        else:
            #print(input, out)
            #print(prob_on[0].item())
            self.heater_on_out = out[0] >= 0.5 #out[0].item() >= out[1].item()


        if self.has_learnt and self.loss is None:
            self.loss = torch.Tensor([-self.threshold])

        if self.loss is not None and not isinstance(self.loss, float):
            self.loss = self.loss.item()

        # TODO do something better
        if ((time >= 3000 and self.loss <= self.threshold) or time >= 9000) and not self.has_learnt:
            print("=========================> DONE @ ", time)
            self.has_learnt = True
            self.model_training = False

    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(current_time)

        return Fmi2Status.ok
    
    def terminate(self) -> int:
        # Save to disk only if we didn't load from disk
        if self.SAVE_TO_DISK and not (self.model_load_path and self.model_load_path.strip()):        
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            
            # Save the model state dict
            model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_model.pt")
            print(f"Saving to {model_path}...")
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Also save the entire model
            full_model_path = os.path.join(self.SAVE_DIR, "thermostat_nn_full_model.pt")
            torch.save(self.model, full_model_path)
            print(f"Full model saved to {full_model_path}")
            
        return Fmi2Status.ok
