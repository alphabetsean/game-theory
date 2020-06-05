import numpy as np
import matplotlib.pyplot as plt

def save_fig(ode_res, time_points, labels, show = False, percent = False, **kwargs):
    """Save the figure provided by the ODE result

    Parameters
    ----------
    ode_res : array
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html for more details
    show : bool
        Whether or not to show the figure
    label : list
        The label names
    """
    fig_name = kwargs.get('fig_name', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    # fig, ax = plt.subplots()

    # Formatting
    # plt.yscale('log')
    
    # plot the fraction of the population
    if percent:
        plt.ylim(0,1)
        # get fractions
        res_sum_column = np.sum(ode_res, axis=1)
        percentage = np.divide(ode_res, res_sum_column[:, None])
        
        ys = percentage[:,].T
    else:
        ys = ode_res[:,].T

    for y, label in zip(ys, labels):
        plt.plot(time_points, y, label=label)

    # set labels
    if xlabel: plt.xlabel(xlabel)
    else: plt.xlabel(r'Time')
    if xlabel: plt.xlabel(xlabel)
    else: plt.ylabel(r'Fraction')
    
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.margins(0,0)

    # saving fig
    if fig_name:
        plt.savefig(fig_name, dpi=300, bbox_inches = 'tight', pad_inches = 0)

    # show fig
    if show: plt.show()

    plt.close('all')