def sir(sir, t, paras):
    """ODE for SIR Model

    Parameters
    ----------
    sir : set
        A set for population density of each type.
        (S, I, R, D, Imu)
    t : array
        The array for the time steps for ODE solver
    paras : array
        The array for the optional parameters for SIR system
    """

    # get rate parameters as time dependent
    try:
        beta = paras['beta']
        gamma = paras['gamma']
        mu = paras['mu']
    except:
        beta, gamma, mu = paras

    dsdt = - (beta * sir[0] * sir[1])
    didt = (beta * sir[0] * sir[1]) - gamma * sir[1] - mu * sir[1]
    drdt = gamma * sir[1]
    dddt = mu * sir[1]
    dIdt = beta * sir[0] * sir[1]

    dsirdt = [dsdt, didt, drdt, dddt, dIdt]

    return dsirdt



def sirv(sir, t, paras):
    """ODE for SIR Model with vaccination

    Parameters
    ----------
    sir : set
        A set for population density of each type.
        (S, I, R, V)
    t : array
        The array for the time steps for ODE solver
    paras : array
        The array for the optional parameters for SIR system
    """
    try:
        beta = paras['beta']
        gamma = paras['gamma']
        q = paras['q']
        qv = paras['qv']
        vac = paras['vac']
    except:
        beta, gamma, q, qv, vac = paras

    dsdt = - (beta * sir[0] * sir[1]) + qv * sir[3] + q * sir[2]
    didt = (beta * sir[0] * sir[1]) - gamma * sir[1]
    drdt = gamma * sir[1] - q * sir[2]
    dvdt = vac * sir[0] * sir[0] - qv * sir[3]

    dsirvdt = [dsdt, didt, drdt, dvdt]

    return dsirvdt
