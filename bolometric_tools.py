"""
Tools to construct bolometric luminosity written by Kaew Tinyanont.
Some of these tools are copied from WIRC+Pol DRP, but are
reproduced here to remove dependency. 
"""
from astropy.io import fits, ascii as asci
import astropy.constants as const
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

import extinction
import george

###########BLACK BODY FIT#################
####My bolometric luminosity
#constants, all cgs

#constants, all cgs
# h = 6.6260755e-27
# kB = 1.380658e-16
# c = 2.99792458e10
# stefan_boltzmann = 5.67051e-5
h = const.h.to(u.erg*u.s).value
kB = const.k_B.to(u.erg/u.K).value
c = const.c.to(u.cm/u.s).value
stefan_boltzmann = const.sigma_sb.to(u.erg/u.s/u.K**4/u.cm**2).value

#Define Planck function. Move to astropy at some point
def B_nu(nu, T):
    return 2*h*nu**3/c**2/(np.exp(h*nu/kB/T) - 1)

#Black body flux at a distance dist, fit this function to get T and r
def BB_flux(wl, T, r, dist):
    """Fit blackbody curve to data"""
    nu = c/(wl/1e8) ###########Angstrom
    flux = np.pi*B_nu(nu, T)*(r**2/dist**2)
    return 1e23*flux #Jansky here

#Actually fit the photometry
def fit_blackbody_curvefit(photometry, distance = 10*u.Mpc.to(u.cm).value, EB_V = 0, RV = 3.1, use_extrapolation = False, plot = False,
                          try_ignore_r_band = True, phot_to_plot = None):
    """photometry is a list of 2d arrays. Each epoch in photometry is each element. 
    Each element is 3d: 
        first index is the wavelength (in angstrom) 
        second index is the flux in Jansky
        third indes is flux uncertainty
        """
#     T = np.zero(len(photometry))
#     R = np.zero(len(photometry))
    #Define a function to fit: just call BB_flux with fixed distance
    def BB_to_fit(wl, T, r):
        return BB_flux(wl, T, r, distance)
    if phot_to_plot is None:
        phot_to_plot = photometry
    bb_fits = []
    bb_err = []
    print("There are %d epochs to fit."%photometry.shape[0])
    for ind, i in enumerate(photometry):
        print("Fitting epoch %d out of %d."%(ind+1,photometry.shape[0]))
        try:
            wl = i[0]
            flux = i[1]
            eflux = i[2]
            extrapolated = i[3]
    #         print(extrapolated.astype('bool'))
            if use_extrapolation == False:
                wl = wl[~extrapolated.astype('bool')]
                flux = flux[~extrapolated.astype('bool')]
                eflux = eflux[~extrapolated.astype('bool')]
            if try_ignore_r_band:
                if len(wl) >=3:#enough bands, ignore r band because of H alpha
                    r_band = wl == 6156.4
                    wl = wl[~r_band]
                    flux = flux[~r_band]
                    eflux = eflux[~r_band]
            if EB_V != 0:
                flux = extinction.apply(-extinction.fitzpatrick99(wl, RV*EB_V, RV), flux)
    #         print(wl)
            #now run the fit using scipy curve fit.
            bb_fit, bb_conv = curve_fit(BB_to_fit, wl, flux, p0 = [5000, 0.3e17], sigma = eflux, \
                        absolute_sigma = True,maxfev = 10000)
            print(bb_fit)
            print(bb_conv)
            bb_fits += [bb_fit]
            bb_err += [ [np.sqrt(bb_conv[0,0]), np.sqrt(bb_conv[1,1]) ]]

            if plot:
                wl_p = phot_to_plot[ind][0]
                flux_p = phot_to_plot[ind][1]
                eflux_p = phot_to_plot[ind][2]
                extrapolated_p = phot_to_plot[ind][3]
        #         print(extrapolated.astype('bool'))
                if use_extrapolation == False:
                    wl_p = wl_p[~extrapolated_p.astype('bool')]
                    flux_p = flux_p[~extrapolated_p.astype('bool')]
                    eflux_p = eflux_p[~extrapolated_p.astype('bool')]
                if EB_V != 0:
                    flux_p = extinction.apply(-extinction.fitzpatrick99(wl_p, RV*EB_V, RV), flux_p)
                plt.errorbar(wl_p, flux_p, eflux_p, fmt = 'o', mfc = 'None')
                plt.errorbar(wl, flux, eflux, fmt = 'o')
                ww = np.linspace(2000, 13000, 1000)
                plt.plot(ww, BB_to_fit(ww, bb_fit[0], bb_fit[1]))
                plt.show()
        except:
            bb_fits += [[np.nan,np.nan]]
            bb_err  += [[np.nan,np.nan]]
    
    return np.array(bb_fits), np.array(bb_err)


###########GP Tool to interpolate lightcurves#################
def george_lightcurve(mjd_obs, flux_obs, flux_err, x_pred = None, kernel_name = 'ExpSquared', 
                      timescale = 100, mean = 'zero', plot = True):
    """
    A function to interpolate observed light curve onto some common time grid "x_pred"
    mjd_obs:    array of the observed mjd
    flux_obs:   array of observed flux. Note that this can be mag, but it works best in the flux space
    flux_err:   array of flux uncertainties 
    x_pred:     This is the array of some common mjd grid to interpolate onto
    kernel_name: Either ExpSquared (george.kernels.ExpSquaredKernel) 
                    or Matern (george.kernels.Matern32Kernel)
    timescale:  length scale of the interpolation
    mean:       mean of the kernel. This should be "zero" for well-sampled light curve
                    and "mean" if there's only a few epochs (rise and fall not covered)
    plot:       whether to produce plots
    """
    x = mjd_obs
    y = flux_obs
    yerr = flux_err

    #Define Kernel
    if kernel_name == 'ExpSquared':
        if mean == 'zero':
            kernel = np.var(y) * george.kernels.ExpSquaredKernel(timescale)
        elif mean == 'mean':
            kernel = np.var(y) * george.kernels.ExpSquaredKernel(timescale) + np.mean(y)
    elif kernel_name == 'Matern':
        if mean == 'zero':
            kernel = np.var(y) * george.kernels.Matern32Kernel(timescale)
        elif mean == 'mean':
            kernel = np.var(y) * george.kernels.Matern32Kernel(timescale) + np.mean(y)
    gp = george.GP(kernel)
    #Compute
    gp.compute(x, yerr)

    #Compute prediction
    if x_pred is None:
        x_pred = np.linspace(np.min(x), np.max(x), 500)
    pred, pred_var = gp.predict(y, x_pred, return_var=True)

    #Plot
    if plot:
        plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                        color="k", alpha=0.2)
        plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
        plt.errorbar(x, y, yerr=yerr, fmt=".g", capsize=0)
        # plt.plot(x_pred, np.sin(x_pred), "--g")
        # plt.xlim(0, 10)
        # plt.ylim(-1.45, 1.45)
        plt.xlabel("MJD")
        plt.ylabel("flux")

    # plt.ylim([20,16])
    
    #Extrapolated flag: True for x_pred < min(mjd_obs) or x_pred > max(mjd_obs)
    extrapolated = np.logical_or(x_pred < np.min(mjd_obs)-1, x_pred > np.max(mjd_obs)+3)
    return x_pred, pred, pred_var, gp, extrapolated
