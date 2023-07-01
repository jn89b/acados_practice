#usr/bin/env python3
from acados_template import AcadosModel, AcadosOcp, \
    AcadosOcpSolver, AcadosSimSolver
# from aircraft_model import aircraft_model
from simple_car_model import CarModel
import scipy.linalg
import numpy as np
import time, os
import matplotlib.pyplot as plt
import Config
import random

# obs_x = 30
# obs_y = 20

seed_val = 15
random.seed(seed_val)

obstacle_array = []
for i in range(Config.N_OBSTACLES):
    #random number between -100 and 100
    obs_x = random.randint(Config.X_MIN, Config.X_MAX)
    obs_y = random.randint(Config.Y_MIN, Config.Y_MAX)
    obstacle_array.append((obs_x, obs_y))
    # obstacle_array.append(obs_y)

#turn obstacles into 1d array
obstacles = np.array(obstacle_array).flatten()
print(obstacles)

yref = np.array([Config.GOAL_X, Config.GOAL_Y, np.deg2rad(270), 0, 0])
yref_e = np.array([Config.GOAL_X, Config.GOAL_Y, np.deg2rad(270)])

class AcadosSettings():
    """
    Acados Settings formulates the OCP solver by 
    Refer to the following documentation for more information
    https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    """
    def __init__(self, model) -> None:
       
        #initialize solver
        self.ocp = AcadosOcp()

        #import model with constraints in the system
        self.model = model
        
        #define acados ODE solver
        self.model_ac = AcadosModel()
        self.init_model()
        self.set_dimensions()
        self.set_cost_matrix()
        self.set_constraints()
        self.set_slack_variables()     
        self.set_prediction_horizon()
        self.set_solver_options()

        self.set_initial_conditions()
        self.set_terminal_conditions()

    def init_model(self)->None:
        """
        Initialize the model formualtion for the OCP 
        To do this we need to the following:
        - define the implicit and explicit expressions
        - define the states, inputs, outputs, and parameters
        - define the name of the model
        - define the model for the OCP

        should do this with keyword arguments from model side
        """
        self.model_ac.f_impl_expr = self.model.f_impl_expr
        self.model_ac.f_expl_expr = self.model.f_expl_expr
        self.model_ac.con_h_expr = self.model.con_h_expr #constraint expression
        self.model_ac.con_h_expr_e = self.model.con_h_expr_e #constraint expression at end of horizon   
        self.model_ac.p = self.model.p #parameters for obstacle avoidance
        self.model_ac.npa = self.model.p.size()[0] #number of parameters
        self.model_ac.x = self.model.states
        self.model_ac.xdot = self.model.z_dot
        self.model_ac.u = self.model.controls
        self.model_ac.name = self.model.name

        #feed this all into model
        self.ocp.model = self.model_ac
        self.ocp.dims.np = self.model.p.size()[0] #number of parameters

    def set_constraints(self)->None:
        """
        Define the constraints for the OCP solver
        Where the constraints are defined in the model
        """
        # self.ocp.constraints.idxbu = np.array([0, 1])
        for key, value in self.model.constraints.items():
            if value is not None:
                setattr(self.ocp.constraints, key, value)


    def set_slack_variables(self) -> None:
        """
        function to set the slack variables for the OCP solver
        ns is the number of slack variables
        nsh is the number of soft constraints 

        """
        for key,value in self.model.slack_variables.items():
            if value is not None:
                setattr(self.ocp, key, value)
        
        # nsh = Config.N_OBSTACLES
        # self.ocp.constraints.lsh = np.zeros(nsh)
        # self.ocp.constraints.ush = np.ones(nsh) 
        # self.ocp.constraints.idxsh = np.array(range(nsh))

        # ns = Config.N_OBSTACLES
        # self.ocp.cost.zl = 10e3 * np.ones((ns,)) # gradient wrt lower slack at intermediate shooting nodes (1 to N-1)
        # self.ocp.cost.Zl = 1 * np.ones((ns,))    # diagonal of Hessian wrt lower slack at intermediate shooting nodes (1 to N-1)
        # self.ocp.cost.zu = 0 * np.ones((ns,))    
        # self.ocp.cost.Zu = 1 * np.ones((ns,))  
        

    def set_dimensions(self)->None:
        """
        Set the dimensions of the model
        Where 
        nx is the number of states
        nu is the number of inputs
        ny is the number of outputs
        ny_e is the number of outputs at the end of the horizon 
        for this formulation ny_e = nx, that is 
        state at the end of the horizon is the same as the state
        N is the number of discretization points, or our prediction horizon
        """
        self.nx = self.model.states.size()[0]
        self.nu = self.model.controls.size()[0]
        self.ny = self.nx + self.nu #total number of states
        self.ny_e = self.nx   

    def set_cost_matrix(self):
        """
        Set the cost function for the OCP solver
        Q is the weighting matrix for the states
        R is the weighting matrix for the inputs
        Qe is terminal error weighting matrix
        The higher the value of Q the more important the for the state
        The higher the value of R the more penalty for the that input
        Refer to the documentation for more information for the ocp solver
        https://docs.acados.org/python_interface/
        """
        #create a diagonal matrix based on size of the state
        #should have model define this
        self.Q = np.diag([0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.Qe = np.diag([500, 500, 500])
        
        #catch error if Q does not equal to the number of states
        # if self.Q.shape[0] != self.nx:
        #     raise Exception(
        #         "Q must be the same size as the number of states")
        # #catch error if R does not equal to the number of inputs
        # if self.R.shape[0] != self.nu:
        #     raise Exception(
        #         "R must be the same size as the number of inputs")
        # if self.Qe.shape[0] != self.ny_e:
        #     raise Exception(
        #         "Qe must be the same size as the number of states at the end of the horizon")
        
        #send to OCP cost solver
        #intermediate cost
        self.ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx)) #states+controls, states
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx) 
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx : self.nx + self.nu, 0:self.nu] = np.eye(self.nu)
        #terminal cost
        self.ocp.cost.W_e = self.Qe
        self.ocp.cost.Vx_e = np.eye(self.nx)

        #reference error
        self.ocp.cost.yref = yref
        self.ocp.cost.yref_e = yref_e

    def set_prediction_horizon(self):
        # for key, value in self.model.prediction_horizon.items():
        #     if value is not None:
        #         setattr(self.ocp.solver_options, key, value)
        self.ocp.solver_options.tf = self.model.prediction_horizon['tf']
        self.ocp.dims.N = self.model.prediction_horizon['N']

    def set_solver_options(self):
        """
        Set the solver options for the OCP solver
        For more information refer to the documentation
        https://docs.acados.org/c_interface/index.html#docstring-based-documentation
        Solver options are located in the following documentation as follow
        https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        Page 8
        """
        qp_solvers = ['FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_HPIPM',
                      'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP']
        self.ocp.solver_options.qp_solver = qp_solvers[4] #'FULL_CONDENSING_HPIPM'#' #'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK' #explicit runge kutta
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 100   
        #regularize the hessian
        self.ocp.solver_options.levenberg_marquardt = 0.15
        self.ocp.solver_options.tol = 1e-4
        self.ocp.solver_options.qp_solver_iter_max = 100     
        # self.ocp.solver_options.alpha_min = 0.05

    def set_initial_conditions(self) -> None:
        # self.ocp.parameter_values = np.array([obs_x, obs_y]) #obstacle position
        self.ocp.constraints.x0 = self.model.init_x #has to be an array
        self.ocp.parameter_values = obstacles
        print("size of parameter_values", self.ocp.parameter_values.shape)

    def set_terminal_conditions(self) -> None:
        print("size of yref", self.ocp.cost.yref.shape)
        print("size of yref_e", self.ocp.cost.yref_e.shape)

def compute_distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

if __name__ == '__main__':
    plt.close("all")

    car_model = CarModel()
    car_model_ac = AcadosSettings(car_model)

    #begin simulation
    ocp = car_model_ac.ocp
    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file = 'acados_ocp' + ocp.model.name +'.json')
    print("good to go!")

    acados_integrator = AcadosSimSolver(
        ocp, json_file = 'acados_ocp' + ocp.model.name +'.json')
    
    # prepare simulation
    Nsim = 1000
    N_horizon = ocp.dims.N
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim, nu))
    time_array = []

    xy_predictions = np.zeros((N_horizon,2))    
    #initial and reference conditions
    X0 = np.array([0, 0, np.deg2rad(0)])
    terminal_tolerance = 0.5

    tcomp_sum = 0
    tcomp_max = 0
    xcurrent = X0
    simX[0, :] = xcurrent
    print("starting at ", xcurrent)
    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
        acados_ocp_solver.set(stage,"p", obstacles)

    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
        acados_ocp_solver.set(stage,"p", obstacles)
        xy_predictions[stage, :] = xcurrent[0:2]

    # closed loop
    for i in range(Nsim):

        #check if we are done
        finished_iteration = i
        if np.linalg.norm(xcurrent[0:2] - yref[0:2]) < terminal_tolerance:
            print("reached target")
            print("heading is ", np.rad2deg(xcurrent[2]));
            break

        #check if in obstacle
        for obst in obstacle_array:
            obst = np.array(obst)
            if (compute_distance(xcurrent[0], xcurrent[1], obst[0], obst[1]) <= Config.OBSTACLE_RADIUS):
                
                #plot the solution that failed
                fig,ax = plt.subplots()
                ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "bo")
                ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "b")
                
                #plot the projections
                ax.plot(xy_predictions[:, 0], xy_predictions[:, 1], "go")

                for obst in obstacle_array:
                    ax.add_patch(plt.Circle((obst[0], obst[1]), Config.OBSTACLE_RADIUS, color='r'))
                ax.plot(yref[0], yref[1], "ro")
                ax.plot(yref[0], yref[1], "r")
                plt.show()
                
                dist = compute_distance(xcurrent[0], xcurrent[1], obst[0], obst[1])
                print("obstacle at ", obst)
                raise Exception(
                    f"Current position {xcurrent[0:2]}, distance to obstacle is {dist}"
                )

            # print(np.linalg.norm(xcurrent[0:2] - obst[0:2]))
            # if np.linalg.norm(xcurrent[0:2] - obst[0:2]) < 0.1:
            #     print("in obstacle")
            #     break

        # set initial state constraint
        #wrap heading between 0 and 2pi
        xcurrent[2] = xcurrent[2] % (2 * np.pi)
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # update yref
        for j in range(N_horizon):
            yref = yref
            yref[2] = xcurrent[2]
            acados_ocp_solver.set(j, "yref", yref)
            acados_ocp_solver.set(j,"p", obstacles)
            xy_predictions[j,0] = acados_ocp_solver.get(j, "x")[0] #x1
            xy_predictions[j,1] = acados_ocp_solver.get(j, "x")[1] #x2

            # acados_ocp_solver.set(j,"p", np.array([obs_x, obs_y]))
        yref_N = yref_e
        acados_ocp_solver.set(N_horizon, "yref", yref_N)
        # acados_ocp_solver.set(N_horizon,"p", np.array([obs_x, obs_y]))

        # solve ocp
        t = time.time()
        status = acados_ocp_solver.solve()
        elapsed = time.time() - t
        time_array.append(elapsed)

        # if status not in [0, 2]:
        #     acados_ocp_solver.print_statistics()
        #     #plot the solution that failed
        #     fig,ax = plt.subplots()
        #     ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "bo")
        #     ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "b")
        #     for obst in obstacle_array:
        #         ax.add_patch(plt.Circle((obst[0], obst[1]), Config.OBSTACLE_RADIUS, color='r'))
        #     ax.plot(yref[0], yref[1], "ro")
        #     ax.plot(yref[0], yref[1], "r")

        #     #plot controls
        #     fig1,ax1 = plt.subplots()
        #     ax1.plot(simU[:finished_iteration, 0], "bo", label="velocity")
        #     # ax1.plot(simU[:finished_iteration, 0], "b")
        #     ax1.plot(simU[:finished_iteration, 1], "go", label="steering")
        #     # ax1.plot(simU[:finished_iteration, 1], "g")
        #     ax1.legend()    
        #     plt.show()
        #     dist = np.sqrt((xcurrent[0] - obs_x)**2 + (xcurrent[1] - obs_y)**2)
        #     raise Exception(
        #         f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        #     )

        # if status == 2:
        #     acados_ocp_solver.print_statistics()
        #     dist = np.sqrt((xcurrent[0] - obs_x)**2 + (xcurrent[1] - obs_y)**2)
        #     print(
        #         f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        #     )
        simU[i, :] = acados_ocp_solver.get(0, "u")

        # simulate system
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", simU[i, :])

        status = acados_integrator.solve()
        # update state
        xcurrent = acados_integrator.get("x")
        simX[i + 1, :] = xcurrent
        print("current state", xcurrent)

    acados_ocp_solver.print_statistics()
    print("average time", np.mean(time_array))
    # plot results
    # if finished_iteration is None:
    #     finished_iteration = Nsim


    fig,ax = plt.subplots()
    ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "bo")
    ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "b")
    ax.plot(yref[0], yref[1], "ro")
    ax.plot(yref[0], yref[1], "r")

    #draw obstacles based on radius 
    # obs_x = 50
    # obs_y = 50
    # obs_r = 10
    # obs = plt.Circle((obs_x, obs_y), obs_r, color='r')

    # ax = plt.gca()
    # ax.add_artist(obs)
    for obst in obstacle_array:
        ax.add_patch(plt.Circle((obst[0], obst[1]), Config.OBSTACLE_RADIUS, color='r'))

    #set label
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    # plt.show()

    #plot controls
    fig1,ax1 = plt.subplots()
    ax1.plot(simU[:finished_iteration, 0], "bo", label="velocity")
    # ax1.plot(simU[:finished_iteration, 0], "b")
    ax1.plot(simU[:finished_iteration, 1], "go", label="steering")
    # ax1.plot(simU[:finished_iteration, 1], "g")
    ax1.legend()    

    plt.show()



        
