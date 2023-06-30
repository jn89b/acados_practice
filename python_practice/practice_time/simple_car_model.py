import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

class CarModel():
    def __init__(self, 
                 time_horizon:float=2.0,
                 num_steps:int=50) -> None:
        self.name = "car_model"
        self.define_states()
        self.define_controls()
        self.define_state_space()
        self.define_constraints()
        self.define_slack_variables()
        self.define_prediction_horizon(time_horizon,num_steps)
        self.init_x = np.array([0.0, 0.0, 0.0])

    def define_states(self) -> None:
        self.x = SX.sym("x")
        self.y = SX.sym("y")
        self.psi = SX.sym("psi")
        
        self.states = vertcat(
            self.x, self.y, self.psi)

    def define_controls(self) -> None:
        self.v_cmd = SX.sym("v_cmd")
        self.psi_cmd = SX.sym("psi_cmd")
        
        self.controls = vertcat(
            self.v_cmd, self.psi_cmd)

    def define_state_space(self) -> None:
        self.x_dot = SX.sym("x_dot")
        self.y_dot = SX.sym("y_dot")
        self.psi_dot = SX.sym("psi_dot")
        self.z_dot = vertcat(
            self.x_dot, self.y_dot, self.psi_dot)

        x_dot = self.v_cmd * cos(self.psi)
        y_dot = self.v_cmd * sin(self.psi)
        psi_dot = self.psi_cmd
        #set dynamics 
        #explicit
        self.f_expl_expr = vertcat(
            x_dot, y_dot, psi_dot)
        
        #implicit
        self.f_impl_expr = self.z_dot - self.f_expl_expr

        #refactor this? 
        self.p = []

    def define_constraints(self) -> None:
        self.constraints = {
            'lbx': None,
            'ubx': None,
            'lbu': np.array([-15, -np.deg2rad(30)]), #velocity and steering rate
            'ubu': np.array([15 , np.deg2rad(30)])  #velocity and steering rate
        }

    def define_slack_variables(self) -> None:
        self.slack_variables = {
            'zl': None,
            'zu': None,
            'Zl': None,
            'Zu': None
        }

    def define_prediction_horizon(self, time_horizon:float, num_steps:int) -> None:
        self.prediction_horizon = {
            'tf': time_horizon,
            'N': num_steps,
        }


        

    

