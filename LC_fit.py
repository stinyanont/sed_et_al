import numpy as np
from scipy import optimize 

def emp_lightcurve(t, y0, m, t0, g0, sig0, tau, theta):
    """
    return an empirical light curve (in mag) from Vacca & Leibundgent (1996) 
    with three components: Early exponential rise, Gaussian peak, and linear decline
    Input:
        t   -   input time
        y0  -   linear portion intercept (corresponds to MNi)
        m   -   linear slope (fraction of decay energy thermalized)
        t0  -   peak time
        g0  -   peak brightness
        sig0 -  width of the Gaussian peak
        tau -   explosion time
        theta - characteristic time of the rise
    Output:
        fitted light curve in mag
    """
    linear_term = y0 + m*(t-t0)
    gaussian_peak = g0 * np.exp( - (t - t0)**2/(2*sig0**2))
    exp_rise = 1 - np.exp( (tau - t)/theta)
    return (linear_term + gaussian_peak)/exp_rise

def emp_lightcurve_each_term(t, y0, m, t0, g0, sig0, tau, theta):
    """
    return an empirical light curve (in mag) from Vacca & Leibundgent (1996) 
    with three components: Early exponential rise, Gaussian peak, and linear decline
    Input:
        t   -   input time
        y0  -   linear portion intercept (corresponds to MNi)
        m   -   linear slope (fraction of decay energy thermalized)
        t0  -   peak time
        g0  -   peak brightness
        sig0 -  width of the Gaussian peak
        tau -   explosion time
        theta - characteristic time of the rise
    Output:
        fitted light curve in mag
    """
    linear_term = y0 + m*(t-t0)
    gaussian_peak = g0 * np.exp( - (t - t0)**2/(2*sig0**2))
    exp_rise = 1 - np.exp( (tau - t)/theta)
    return linear_term, gaussian_peak, exp_rise


def empirical_fit_LC(t, mag, mag_err):
    """
    Take a light curve in a given band (or bolometric) and fit it with
    emp_lightcurve to find all the fit parameters. Will implement this with 
    emcee one day.
    Input:
        t       -   observation epoch, in rest day
        mag     -   obs magnitude
        mag_err -   mag uncertainty
    Output: 
        fitted parameters y0, m, t0, g0, sig0, tau, theta
    """
    