- [ ] Make a simple shell script to install acados
- [ ] Make a docker file to install acados and with ROS as well
- [ ] Recreate the fixed wing simulations and compare run time between the two 
  - [ ] One not compiled
  - [ ] One with compiled 
- [ ] Figure out how to interact with ROS2 framework 

## Good code references 
For ROS integration:
https://github.com/ivanacollg/MPC_CollisionAvoidance/tree/main

## Setting up constraints properly 
https://discourse.acados.org/t/constraints-and-parameters-for-specific-time-steps-external/305

https://github.com/uzh-rpg/data_driven_mpc/blob/main/ros_gp_mpc/src/quad_mpc/quad_3d_optimizer.py

https://discourse.acados.org/t/how-to-describe-a-geometric-constraints-us-eing-polytopic-formulation/984

https://discourse.acados.org/t/obstacle-avoidance-python-solver-status-2-and-4/648

https://discourse.acados.org/t/is-there-some-memory-in-the-solver/638

## Important notes
USE SQP_RTI or sequential programming for real time integration

## Obstacle 
https://discourse.acados.org/t/parameteric-ocp-with-path-constraint-and-ls-cost/98/2

https://discourse.acados.org/t/infeasibility-issues-when-using-hard-nonlinear-constraints/1021

## Formulating the obstacle avoidance problem
- Introduce a parameter $p$, where p is the distance to the obstacle
- Introduce an