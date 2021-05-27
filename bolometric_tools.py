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
