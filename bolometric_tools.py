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
from scipy.integrate import simpson

import extinction
import george

import sys

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

def B_lam(wl, T):
    return 2*h*c**2/wl**5/(np.exp(h*c/wl/kB/T) - 1)

#Black body flux at a distance dist, fit this function to get T and r
def BB_flux(wl, T, r, dist):
    """Fit blackbody curve to data"""
    nu = c/(wl/1e8) ###########Angstrom
    flux = np.pi*B_nu(nu, T)*(r**2/dist**2)
    return 1e23*flux #Jansky here

#Black body flux at a distance dist, fit this function to get T and r
def BB_flux_lam(wl, T, r, dist):
    """Fit blackbody curve to data"""
    wl = (wl*u.angstrom).to(u.cm).value
    flux = np.pi*B_lam(wl, T)*(r**2/dist**2)
    return flux #erg/s/cm^2/cm here

#Actually fit the photometry
def fit_blackbody_curvefit(photometry, distance = 10*u.Mpc.to(u.cm), z = 0, 
                            EB_V = 0, RV = 3.1, extinction_source = 'MW',
                            use_extrapolation = False, plot = False,
                            suppress_uv = False, suppression_index = 1, cutoff_wavelength = 3000,
                            return_uv_ir = False, uv_cut = 3000, ir_cut = 10000,
                            try_ignore_r_band = True, phot_to_plot = None, epochs = None):
    """
    photometry is a list of 2d arrays. Each epoch in photometry is each element. 
    Each element is 3d: 
        first index is the wavelength (in angstrom) 
        second index is the flux in Jansky
        third indes is flux uncertainty
    distance:               Distance to the source in cm
    z:                      Redshift of the source. Note that this only shifts the observed wavelength and not
                                performing the full K correction
    EB_V:                   The parameter E(B-V)
    RV:                     The parameter RV (= AV/E(B-V))
    extinction_source:      Either "host" or "MW". This way, the dust correction is applied at a correct redshift.
    use_extrapolation:      Whether or not to use extrapolated photometry
    plot:                   To plot or not to plot?
    suppress_uv:            Whether to suppress the UV flux with a factor (lambda/cutoff_wavelength)**suppression_index
                                where lambda < cutoff_wavelength
    cutoff_wavelength:      Cutoff wavelength shorter than which the UV suppression factor applies.
    try_ignore_r_band:      Whether to ignore r band if there are at least two more bands. This is for SN II
                                which has strong H alpha
    phot_to_plot:           optional photometry array to plot along, but not fit. 
        """
#     T = np.zero(len(photometry))
#     R = np.zero(len(photometry))
    #Define a function to fit: just call BB_flux with fixed distance
    if suppress_uv:
        def BB_to_fit(wl, T, r):
            BB_no_suppression =  BB_flux(wl, T, r, distance)
            uv_suppression_factor = (wl/cutoff_wavelength)**(suppression_index)
            uv_suppression_factor[wl >= cutoff_wavelength] = 1
            return BB_no_suppression * uv_suppression_factor
    else:
        def BB_to_fit(wl, T, r):
            return BB_flux(wl, T, r, distance)
    if phot_to_plot is None:
        phot_to_plot = photometry
    bb_fits = []
    bb_err = []
    print("There are %d epochs to fit."%photometry.shape[0])
    for ind, i in enumerate(photometry):
        print("Fitting epoch %d out of %d."%(ind+1,photometry.shape[0]))
        if epochs is not None:
            print("%d day"%epochs[ind])
        try:
            wl = i[0]/(1+z) #correct for redshift here
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
                if extinction_source == 'MW':
                    flux = extinction.apply(-extinction.fitzpatrick99(wl*(1+z), RV*EB_V, RV), flux)
                elif extinction_source == 'host':
                    flux = extinction.apply(-extinction.fitzpatrick99(wl, RV*EB_V, RV), flux)
    #         print(wl)
            #now run the fit using scipy curve fit.
            bb_fit, bb_conv = curve_fit(BB_to_fit, wl, flux, p0 = [5000, 0.3e17], sigma = eflux, \
                        absolute_sigma = True,maxfev = 10000)
            # print(bb_fit)
            # print(bb_conv)
            # bb_fits += [bb_fit]
            # bb_err += [ [np.sqrt(bb_conv[0,0]), np.sqrt(bb_conv[1,1]) ]]
            T_bb = bb_fit[0]
            r_bb = bb_fit[1]
            dT_bb = np.sqrt(bb_conv[0,0])
            dr_bb = np.sqrt(bb_conv[1,1])
            ##################Compute Luminosity########################
            lam = np.linspace(100000, 10, 10000)*1e-8 #hopefully not too fine, cm
            if suppress_uv == False:
                #no UV suppression, L is just the stefan boltzmann Luminosity
                L = 4 * np.pi * r_bb**2 * stefan_boltzmann * T_bb**4
                # dL = 4*np.pi*stefan_boltzmann* np.sqrt((2*bb_fits[:,1]*bb_fits[:,0]**4 *bb_errs[:,1])**2 +\
                #                              (4*bb_fits[:,1]**2*bb_fits[:,0]**3 *bb_errs[:,0])**2)
                dL = 4*np.pi*stefan_boltzmann* np.sqrt((2*r_bb*T_bb**4 *dr_bb)**2 +\
                                                (4*r_bb**2*T_bb**3 *dT_bb)**2)
            else: #If UV is suppressed, we have to actually integrate
                nu = c/lam #Hz
                F_nu = np.pi*B_nu(nu, T_bb) #flux at the source
                F_up = np.pi*B_nu(nu, T_bb+dT_bb)
                F_lo = np.pi*B_nu(nu, T_bb-dT_bb)
                uv_suppression_factor = (lam*1e8/cutoff_wavelength)**(suppression_index)
                uv_suppression_factor[lam*1e8 >= cutoff_wavelength] = 1
                # print(uv_suppression_factor)
                L = 4* np.pi * r_bb**2 * simps(F_nu * uv_suppression_factor, nu)
                L_up = 4* np.pi * (r_bb+dr_bb)**2 * simps(F_up * uv_suppression_factor, nu)
                L_lo = 4* np.pi * (r_bb-dr_bb)**2 * simps(F_lo * uv_suppression_factor, nu)
                dL = np.abs(np.mean( [(L_up - L), (L - L_lo)] ))
            #########Now, option to compute UV and IR fluxes
            if return_uv_ir:
                lam_UV = np.linspace(uv_cut, 10, 10000)*1e-8
                lam_IR = np.linspace(100000, ir_cut, 10000)*1e-8
                if suppress_uv:
                    uv_suppression_factor = (lam*1e8/cutoff_wavelength)**(suppression_index)
                    uv_suppression_factor[lam*1e8 >= cutoff_wavelength] = 1
                else:
                    uv_suppression_factor = (lam*1e8/cutoff_wavelength)**(suppression_index)
                    uv_suppression_factor[lam*1e8 >= 0] = 1                   
                nuUV = c/lam_UV #Hz
                nuIR = c/lam_IR #Hz
                F_nu_UV = np.pi*B_nu(nuUV, T_bb) #flux at the source
                F_nu_IR = np.pi*B_nu(nuIR, T_bb) #flux at the source
                # print(uv_suppression_factor)
                L_UV = 4* np.pi * r_bb**2 * simps(F_nu_UV* uv_suppression_factor, nuUV)
                L_IR = 4* np.pi * r_bb**2 * simps(F_nu_IR* uv_suppression_factor, nuIR)

                bb_fits += [[T_bb, r_bb, L, L_UV/L, L_IR/L]]
                bb_err += [[dT_bb, dr_bb, dL]]
            else:
                bb_fits += [[T_bb, r_bb, L]]
                bb_err += [[dT_bb, dr_bb, dL]]

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
                uv_suppression_factor = np.ones(len(ww))
                if suppress_uv:
                    uv_suppression_factor[ww < cutoff_wavelength] = \
                        (ww[ww < cutoff_wavelength]/cutoff_wavelength)**suppression_index
                plt.plot(ww, BB_to_fit(ww, bb_fit[0], bb_fit[1])/uv_suppression_factor, '--')
                plt.plot(ww, BB_to_fit(ww, bb_fit[0], bb_fit[1]), '-')
                print('plotting')
                plt.show()
        except ValueError as e:
            # # print('bad')
            # print(e)
            bb_fits += [[np.nan,np.nan, np.nan]]
            bb_err  += [[np.nan,np.nan, np.nan]]
    
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
    extrapolated = np.logical_or(x_pred < np.min(mjd_obs)-5, x_pred > np.max(mjd_obs)+5)
    return x_pred, pred, pred_var, gp, extrapolated
