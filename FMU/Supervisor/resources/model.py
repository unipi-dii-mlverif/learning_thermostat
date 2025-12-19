import pickle
from fmi2 import Fmi2FMU, Fmi2Status

ANOMALY_THRESHOLD = 3.0


class Model(Fmi2FMU):
    def __init__(self, reference_to_attr=None) -> None:
        super().__init__(reference_to_attr)
        # Parameters for initial values
        self.C_init = 30.0
        self.H_init = 20.0
        self.UL_init = 0.0
        self.LL_init = 10.0
        # Output variables
        self.LL_out = 10.0
        self.UL_out = 0.0
        self.H_out = 20.0
        self.C_out = 30.0
        # Input variables
        self.heater_on_in = False
        self.T_heater_in = 0.0
        self.T_bair_in = 0.0
        self.T_room_in = 0.0
        self.T_bair_plant_in = 0.0

    def do_step(self, current_time, step_size, no_step_prior):
        # Initialize outputs using parameter values
        self.LL_out = self.LL_init
        self.UL_out = self.UL_init
        self.H_out = self.H_init
        self.C_out = self.C_init
        # Initialize inputs
        self.heater_on_in = False
        self.T_heater_in = 0.0
        self.T_bair_in = 0.0
        self.T_room_in = 0.0
        self.T_bair_plant_in = 0.0

        return Fmi2Status.ok


    def ao(self, current_time, step_size, no_step_prior):
        #if abs(self.T_bair_in - self.T_bair_plant_in) > ANOMALY_THRESHOLD:
        #    self.H_out = -1.0
        #else:
        self.H_out = 20.0
        
        return Fmi2Status.ok

