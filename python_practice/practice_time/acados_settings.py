#usr/bin/env python3
from acados_template import AcadosModel, AcadosOcp, \
    AcadosOcpSolver, AcadosSimSolver
# from aircraft_model import aircraft_model
from simple_car_model import CarModel
import scipy.linalg
import numpy as np
import time, os
import matplotlib.pyplot as plt

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
        self.model_ac.x = self.model.states
        self.model_ac.xdot = self.model.z_dot
        self.model_ac.u = self.model.controls
        # self.model_ac.p = self.model.p
        self.model_ac.name = self.model.name
        self.ocp.model = self.model_ac

    def set_constraints(self)->None:
        """
        Define the constraints for the OCP solver
        Where the constraints are defined in the model
        """
        self.ocp.constraints.idxbu = np.array([0, 1])

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
        # self.ns = self.model.ns #review this
        # self.nsh = self.model.nsh #review this      
        

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
        self.Q = np.diag([1e3, 1e3, 1e1])
        self.R = np.diag([1e-1, 1e-2])
        self.Qe = np.diag([1e3, 1e3, 1e-2])
        
        #catch error if Q does not equal to the number of states
        if self.Q.shape[0] != self.nx:
            raise Exception(
                "Q must be the same size as the number of states")
        #catch error if R does not equal to the number of inputs
        if self.R.shape[0] != self.nu:
            raise Exception(
                "R must be the same size as the number of inputs")
        if self.Qe.shape[0] != self.ny_e:
            raise Exception(
                "Qe must be the same size as the number of states at the end of the horizon")
        
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
        self.ocp.cost.yref = np.zeros((self.ny,))
        self.ocp.cost.yref_e = np.zeros((self.ny_e,))

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
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' #'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK' #explicit runge kutta
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 50   

    def set_initial_conditions(self) -> None:
        self.ocp.constraints.x0 = self.model.init_x #has to be an array

    def set_terminal_conditions(self) -> None:
        print("size of yref", self.ocp.cost.yref.shape)
        print("size of yref_e", self.ocp.cost.yref_e.shape)

if __name__ == '__main__':
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

    #initial and reference conditions
    X0 = np.array([0, 0, np.deg2rad(0)])

    yref = np.array([50, 50, np.deg2rad(270), 0, 0])
    yref_e = np.array([50, 50, np.deg2rad(270)])
    terminal_tolerance = 1e-2

    tcomp_sum = 0
    tcomp_max = 0
    xcurrent = X0
    simX[0, :] = xcurrent
    print("starting at ", xcurrent)
    # initialize solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # closed loop
    for i in range(Nsim):

        #check if we are done
        if np.linalg.norm(xcurrent[0:1] - yref[0:1]) < terminal_tolerance:
            finished_iteration = i
            print("reached target")
            print("heading is ", np.rad2deg(xcurrent[2]));
            break

        # set initial state constraint
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # update yref
        for j in range(N_horizon):
            yref = yref
            acados_ocp_solver.set(j, "yref", yref)
        yref_N = yref_e
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp
        print("solve ocp")
        t = time.time()
        status = acados_ocp_solver.solve()
        elapsed = time.time() - t
        time_array.append(elapsed)

        if status not in [0, 2]:
            acados_ocp_solver.print_statistics()
            raise Exception(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )

        if status == 2:
            print(
                f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
            )
        simU[i, :] = acados_ocp_solver.get(0, "u")

        # simulate system
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", simU[i, :])

        status = acados_integrator.solve()
        if status != 0:
            raise Exception(
                f"acados integrator returned status {status} in closed loop instance {i}"
            )

        # update state
        xcurrent = acados_integrator.get("x")
        simX[i + 1, :] = xcurrent
        print("current state", xcurrent)

    acados_ocp_solver.print_statistics()
    print("average time", np.mean(time_array))
    # plot results
    plt.figure()
    plt.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "bo")
    plt.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "b")
    plt.plot(yref[0], yref[1], "ro")
    plt.plot(yref[0], yref[1], "r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


        
