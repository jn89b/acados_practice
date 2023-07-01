from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat
import matplotlib.pyplot as plt


def export_ode_model():
    modelname = 'testmodel'

    # set up states & controls
    x1 = SX.sym('x')