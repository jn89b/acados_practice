import Config
import casadi as ca
import numpy as np

min_obs_dist = Config.OBSTACLE_RADIUS + 2.0
max_obs_dist = 1e5


ROLL_MIN = np.deg2rad(-45)
ROLL_MAX = np.deg2rad(45)
PITCH_MIN = np.deg2rad(-20)
PITCH_MAX = np.deg2rad(20)
YAW_MIN = np.deg2rad(-360)
YAW_MAX = np.deg2rad(360)

V_MIN = 1.0
V_MAX = 10.0
RRATE_MIN = np.deg2rad(-35)
RRATE_MAX = np.deg2rad(35)

PRATE_MIN = np.deg2rad(-15)
PRATE_MAX = np.deg2rad(15)

YRATE_MIN = np.deg2rad(-15)
YRATE_MAX = np.deg2rad(15)

class AircraftModel():
    """
    Aircraft simple model which is used for the MPC

    Args:
        time_horizon (float): Time horizon for the prediction (default: 2.0)
        num_steps (int): Number of prediction steps (default: 40)
    """

    def __init__(self, 
                 time_horizon:float=2.0,
                 num_steps:int=40) -> None:
        self.name = "plane_model"
        self.define_states()
        self.define_controls()
        self.define_state_space()
        self.define_parameters()
        self.define_constraints()
        self.define_slack_variables()
        self.define_prediction_horizon(time_horizon,num_steps)
        self.define_cost()
        self.init_x = np.array([1.0, 1.0, 50.0, 0.05, 0.05, 0.05])

    def define_states(self):
        """define the states of your system"""
        #positions ofrom world
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f
        )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """controls for your system"""
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )

    def define_state_space(self):
        """define the state space of your system"""
        self.g = 9.81 #m/s^2
        self.x_dot = ca.SX.sym('x_dot')
        self.y_dot = ca.SX.sym('y_dot')
        self.z_dot = ca.SX.sym('z_dot')
        self.phi_dot = ca.SX.sym('phi_dot')
        self.theta_dot = ca.SX.sym('theta_dot')
        self.psi_dot = ca.SX.sym('psi_dot')

        self.z_dot = ca.vertcat(
                self.x_dot,
                self.y_dot,
                self.z_dot,
                self.phi_dot,
                self.theta_dot,
                self.psi_dot
            )
 
        #set dynamics 
        #body to inertia frame
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        
        self.phi_fdot = self.u_phi
        self.theta_fdot = self.u_theta
        self.psi_fdot = (self.g * (ca.tan(self.phi_f) / self.v_cmd))

        #explicit dynamics
        self.f_expl_expr = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot
        )

        #implicit dynamics
        self.f_impl_expr = self.z_dot - self.f_expl_expr
        self.cost_expr_y = ca.vertcat(self.states , self.controls)
        self.cost_expr_y_e = ca.vertcat(self.states)

    def define_parameters(self) -> None:
        """
        applying inequality constraints of 
        lh <= p <= uh
        lhe <= pe <= uhe
        """
        n_obstacles = Config.N_OBSTACLES
        self.obs_x = ca.SX.sym('obs_x', n_obstacles)
        self.obs_y = ca.SX.sym('obs_y', n_obstacles)        
        self.dist = np.sqrt(((self.x_f- self.obs_x)*(self.x_f- self.obs_x)) \
                         +((self.y_f - self.obs_y)*(self.y_f - self.obs_y)))

        self.obst_cost = ca.SX.sym('obst_cost', n_obstacles)
        self.sum_cost = ca.sum2(
            (1.0 / ca.exp((self.x_f - self.obs_x) + (self.y_f - self.obs_y)**2))
        )

        self.p = ca.vertcat(self.obs_x, self.obs_y) #parameters for obstacle avoidance constraint
        self.con_h_expr = self.dist #constraint expression
        self.con_h_expr_e = self.dist #constraint expression at end of horizon  

    def define_constraints(self) -> None:
        #boundary constraints
        self.constraints = {
            'lbx': np.array([-1E15, -1E15, 30, ROLL_MIN, PITCH_MIN, YAW_MIN]),
            'ubx': np.array([1E15, 1E15, 60, ROLL_MAX, PITCH_MAX, YAW_MAX]),
            'lbu': np.array([RRATE_MIN, PRATE_MIN, YRATE_MIN, V_MIN]), #p, q, r, v_cmd
            'ubu': np.array([RRATE_MAX, PRATE_MAX, YRATE_MAX, V_MAX]),  #p, q, r, v_cmd
            'idxbx': np.array([0,1,2,3,4,5]),
            'idxbu': np.array([0,1,2,3]),
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

    def define_cost(self):
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([1, 1, 1, 1])
        self.Qe = np.diag([50, 50, 50, 50, 50, 50])
