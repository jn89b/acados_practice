import numpy as np
import Config
# from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, sum2, exp

#create a 1d array of 5 elements with values of 2.0
# x = np.full(5, 2.0)

min_obs_dist = Config.OBSTACLE_RADIUS + 2.0
max_obs_dist = 1e5

class CarModel():
    def __init__(self, 
                 time_horizon:float=2.0,
                 num_steps:int=40) -> None:
        self.name = "car_model"
        self.define_states()
        self.define_controls()
        self.define_state_space()
        self.define_parameters()
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
        self.cost_expr_y = vertcat(self.states , self.controls)
        self.cost_expr_y_e = vertcat(self.states)

    def define_parameters(self) -> None:
        """
        applying inequality constraints of 
        lh <= p <= uh
        lhe <= pe <= uhe
        """
        n_obstacles = Config.N_OBSTACLES
        self.obs_x = SX.sym('obs_x', n_obstacles)
        self.obs_y = SX.sym('obs_y', n_obstacles)        
        self.dist = np.sqrt(((self.x- self.obs_x)*(self.x- self.obs_x)) \
                         +((self.y - self.obs_y)*(self.y - self.obs_y)))

        self.obst_cost = SX.sym('obst_cost', n_obstacles)
        self.sum_cost = sum2(
            (1.0 / exp((self.x - self.obs_x) + (self.y - self.obs_y)**2))
        )

        self.p = vertcat(self.obs_x, self.obs_y) #parameters for obstacle avoidance constraint
        self.con_h_expr = self.dist #constraint expression
        self.con_h_expr_e = self.dist #constraint expression at end of horizon  

    def define_constraints(self) -> None:
        #boundary constraints
        self.constraints = {
            'lbx': None,
            'ubx': None,
            'lbu': np.array([0.5, -np.deg2rad(45)]), #velocity and steering rate
            'ubu': np.array([10.0, np.deg2rad(45)]),  #velocity and steering rate
            'idxbu': np.array([0,1]),
            'uh': np.full(Config.N_OBSTACLES, max_obs_dist), #upper bound on obstacle avoidance constraint
            'uh_e': np.full(Config.N_OBSTACLES, max_obs_dist), #upper bound on obstacle avoidance constraint at end of horizon
            'lh': np.full(Config.N_OBSTACLES, min_obs_dist), #lower bound on obstacle avoidance constraint
            'lh_e': np.full(Config.N_OBSTACLES, min_obs_dist) #lower bound on obstacle avoidance constraint at end of horizon
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


        

    

