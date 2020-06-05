from utils import save_fig
from diff_equations import *
import numpy as np
import scipy as sp
from scipy.integrate import odeint


################################################################################
#                                     SIR                                      #
################################################################################
init = (1 - 1e-6, 1e-6, 0, 0, 0)
time = np.linspace(0, 400, 500)

paras = {
    'beta': 2.91 * (0.0488 + 0.0488 / 8.5),  # infection rate
    'gamma': 0.0488,  # recovery rate
    'mu': 0.0488 / 8.5,
}
res = odeint(sir, init, time, args=(paras,))

label = [r'$S(t)$', r'$i(t)$', r'$R(t)$', r'$D(t)$', r'$I(t)$']
save_fig(res, time, label, percent=True, fig_name="../fig/SIR_percent.png")
save_fig(res, time, label, fig_name="../fig/SIR.png")


################################################################################
#                                     SIRV                                     #
################################################################################
init = (1, 1e-4, 0, 0)
time = np.linspace(0, 400, 500)
paras = {
    'beta': 2.91 * (0.0488 + 0.0488 / 8.5),  # infection rate
    'gamma': 0.0488,  # recovery rate
    'vac': 1e-2,  # rate of vaccination
    'qv': 1e-5,  # rate of becoming S
    'q': 1e-5,  # rate of losing imunity
    'mortality': 1e-2,
}
res = odeint(sirv, init, time, args=(paras,))

label = [r'$S(t)$', r'$i(t)$', r'$R(t)$', r'$V(t)$']
save_fig(res, time, label, percent=True, fig_name="../fig/SIRV_percent.png")
save_fig(res, time, label, fig_name="../fig/SIRV.png")
