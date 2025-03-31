# ToDo define wrapper model for the FMI interface
# Xdata = data[["{Ego}.EgoInstance.ego_velocity","{Ego}.EgoInstance.rel_position","{Ego}.EgoInstance.rel_velocity"]].values
import pickle
import SlidingWindowStepModel
from fmi2 import Fmi2FMU, Fmi2Status
import torch
from torch import nn
import statistics
from SlidingWindowStepModel import *
import random

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
        
        self.T_desired = 35.0
        self.last_time = float("-inf")
        self.threshold = 0.005 #0.001 
        self.model_training = True
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Softmax(dim=-1)
            nn.Sigmoid(),
        )
        self.n_true = 0
        self.n_false = 0

        self.swsm = SWSM(self.model, 40)

    def exit_initialization_mode(self):
        return Fmi2Status.ok

    def ctrl_step(self, time):
        if time < self.last_time + 1:
            return
        
        self.last_time = time

        if not self.has_learnt:
            print(self.loss)

        # Poor men's balancing
        if not self.has_learnt and self.n_false != 0 and not self.heater_on_in and ((self.n_false / (self.n_true + self.n_false) > .4) or (random.random() <= 0.2)):
            print("skipping", self.n_false, self.n_true)
            return
        
        if self.heater_on_in:
            self.n_true += 1
        else:
            self.n_false += 1
        
        input = torch.Tensor([self.T_bair_in, self.H_in, self.T_desired-self.LL_in])
        target = torch.Tensor([1.0 if self.heater_on_in else 0.0]) #, 1.0 if not self.heater_on_in else 0.0])
        #if self.heater_on_in:
        #    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", input, target)
        out, loss2 = self.swsm.accumulate_and_step(input, self.model_training, target)
        if loss2 is not None:
            self.n_false = 0
            self.n_true = 0
            self.loss = loss2
        
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

        if time >= 2000 and self.loss <= self.threshold and not self.has_learnt:
            print("=========================> DONE @ ", time)
            self.has_learnt = True
            self.model_training = False

    def do_step(self, current_time, step_size, no_step_prior):
        self.ctrl_step(current_time)

        return Fmi2Status.ok
