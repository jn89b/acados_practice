import matplotlib.pyplot as plt
import Config
import numpy as np
from acados_template import AcadosOcpSolver, AcadosSimSolver
# from aircraft_model import aircraft_model
from acados_settings import AcadosSettings
from simple_car_model import CarModel
import numpy as np
import time
import matplotlib.pyplot as plt
import random

seed_val = 15
random.seed(seed_val)

obstacle_array = []
for i in range(Config.N_OBSTACLES):
    #random number between -100 and 100
    obs_x = random.randint(Config.X_MIN, Config.X_MAX)
    obs_y = random.randint(Config.Y_MIN, Config.Y_MAX)
    obstacle_array.append((obs_x, obs_y))

#turn obstacles into 1d array
obstacles = np.array(obstacle_array).flatten()
print(obstacles)

yref = np.array([Config.GOAL_X, Config.GOAL_Y, np.deg2rad(270), 0, 0])
yref_e = np.array([Config.GOAL_X, Config.GOAL_Y, np.deg2rad(270)])

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
            
        # set initial state constraint
        #wrap heading between 0 and 2pi
        xcurrent[2] = xcurrent[2] % (2 * np.pi)
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # update yref
        for j in range(N_horizon):
            # yref = yref
            yref[2] = xcurrent[2]
            acados_ocp_solver.set(j, "yref", yref)
            acados_ocp_solver.set(j,"p", obstacles)
            xy_predictions[j,0] = acados_ocp_solver.get(j, "x")[0] #x1
            xy_predictions[j,1] = acados_ocp_solver.get(j, "x")[1] #x2

        yref_N = yref_e
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

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

    fig,ax = plt.subplots()
    ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "bo")
    ax.plot(simX[:finished_iteration, 0], simX[:finished_iteration, 1], "b")
    ax.plot(yref[0], yref[1], "ro")
    ax.plot(yref[0], yref[1], "r")

    for obst in obstacle_array:
        ax.add_patch(plt.Circle((obst[0], obst[1]), Config.OBSTACLE_RADIUS, color='r'))

    #set label
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)

    #plot controls
    fig1,ax1 = plt.subplots()
    ax1.plot(simU[:finished_iteration, 0], "bo", label="velocity")
    # ax1.plot(simU[:finished_iteration, 0], "b")
    ax1.plot(simU[:finished_iteration, 1], "go", label="steering")
    # ax1.plot(simU[:finished_iteration, 1], "g")
    ax1.legend()    
    plt.show()
