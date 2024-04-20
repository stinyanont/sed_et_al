from astropy.io import ascii as asci
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
# print(os.path.dirname(os.path.realpath(__file__)))
script_path = os.path.dirname(os.path.realpath(__file__))

# print(script_path)

#Read dust emissivity files
#######These are lifted from the plots in Fox+2010. Has several grain sizes, but only out to ~15 micron.m
q_si = asci.read(script_path+'/dust_data/dust_emission_efficiency_fox2010/q_silicate.csv', data_start = 2, delimiter = ',')
q_c =  asci.read(script_path+'/dust_data/dust_emission_efficiency_fox2010/q_graphite.csv', data_start = 2, delimiter = ',')

#######These are from Draine (add citations here, but they are in Tinyanont+2019b, the 2017eaw paper.) 
d_q_si = asci.read(script_path+'/dust_data/draine_dust_properties/sil_1e-1micron.dat')
d_q_c  = asci.read(script_path+'/dust_data/draine_dust_properties/gra_1e-1micron.dat')

q_c.sort('col5')

#Define some constants in cgs
h = 6.6260755e-27
kB = 1.380658e-16
c = 2.99792458e10
Mpc = 3.086e18 *10**6 #cm/Mpc

#Define functions to produce SED models from dust properties
def get_Q(comp = 'C', size = 0.1):
    """Look up a table for the emission coefficient Q for a given dust composition and grain size
    This function uses the table from Fox+2010.
    """
    #Pick composition
    if comp == 'C':
        Q_tab = q_c
    elif comp == 'Si':
        Q_tab = q_si
    else:
        print("I don't have data for %s grains. Try C or Si."%comp)
#         break
    #pick size
    if size == 0.001:
        wl = Q_tab['col1']
        Q  = Q_tab['col2']
    elif size == 0.01:
        wl = Q_tab['col3']
        Q  = Q_tab['col4']
    elif size == 0.1:
        wl = Q_tab['col5']
        Q  = Q_tab['col6']       
    elif size == 1:
        wl = Q_tab['col7']
        Q  = Q_tab['col8']
    else:
        print("Only have data for 0.001, 0.01, 0.1, and 1 micron grains")
#         break
        
    #Interpolate the table
    int_Q = interp1d(wl, Q, bounds_error = False)
    return int_Q

# F = M B_nu kappa/d^2

def B_nu(nu, T):
    """
    Planck Function
    Input  
    nu in s^-1 (array OK)
    T in K (one number)
    Returns black body specific intensity in cgs (erg/s/cm^2/Hz/Sr)
    """
    return 2*h*nu**3/c**2/(np.exp(h*nu/kB/T) - 1)

def kappa(wl,comp, a):
    """
    A function to get dust kappa using Fox+2010 table. 
    Inputs:
        wl: wavelength in micron
        comp: either 'C' (for amorphous carbon) or 'Si' for astro Silicate. 
        a: grain radius: 0.001, 0.01, 0.1, or 1
    Output:
        dust kappa in cgs
    """
    if comp == 'C':
        rho = 2.2
    elif comp == 'Si':
        rho = 3
    Q = get_Q(comp, a)
    a_cgs = a*1e-4
    return (3/(4*np.pi*rho*a_cgs**3))*(np.pi*a_cgs**2*Q(wl))

def draine_kappa(wl, comp):
    """get kappa from Draine dust Q. Only support a = 0.1 micron
    Inputs:
        wl: wavelength in micron
        comp: either 'C' (for amorphous carbon) or 'Si' for astro Silicate. 
    Output:
        dust kappa in cgs
    """
    if comp == 'C':
        dQ = d_q_c
        rho = 2.2
    elif comp == 'Si':
        dQ = d_q_si
        rho = 3
    int_dQ = interp1d(dQ['w(micron)'], dQ['Q_abs'], bounds_error = False)
#     rho = 2.5
    a_cgs = 0.1e-4
    return (3/(4*np.pi*rho*a_cgs**3))*(np.pi*a_cgs**2*int_dQ(wl))
    

def dust_flux(wl, T, mass, comp, a,distance, source = 'Draine'):
    """
    return F_nu of dust emission given mass, composition, size, and temperature
    wl is wavelength in micron
    T is temperature in K
    mass is dust mass in solar mass
    a is dust size in micron
    source is between Dwek and Draine
    """
    wl_cgs = wl*1e-4
    nu = c/wl_cgs
    if source == 'Dwek':
        kap = kappa(wl, comp, a)
    elif source == 'Draine':
        kap = draine_kappa(wl, comp)
    mass_cgs = mass*1.99e33 #g/solar mass
    
    return mass_cgs*B_nu(nu,T)*kap/distance**2 #cgs


def dust_flux_nu(nu, T, mass, comp, a, distance, source = 'Draine'):
    """
    return F_nu of dust emission given mass, composition, size, and temperature
    wl is wavelength in micron
    T is temperature in K
    mass is dust mass in solar mass
    a is dust size in micron
    source is between Dwek and Draine
    """
    wl_cgs = c/nu #cm
    #wl_cgs = wl*1e-4
    #nu = c/wl_cgs
    if source == 'Dwek':
        kap = kappa(wl, comp, a)
    elif source == 'Draine':
        kap = draine_kappa(wl, comp)    
        mass_cgs = mass*1.99e33 #g/solar mass
    
    return mass_cgs*B_nu(nu,T)*kap/distance**2 #cgs
    
    
def flux_in_filter( wl, F_nu, filters, FWHM = False):
    """
    This function takes the wavelengths and corresponding flux spectrum (F_nu from dust_flux), and filter
    definition (a list of 2-element lists, denoting cut-on and cut-off wavelengths in micron)
    and return flux expected in those filters. 
    """
    wl_cgs = wl*1e-4 #from micron to cm
    #convert to F_lambda
    F_lam = c/wl_cgs**2 * F_nu
    
    #now, get integrated flux in each band
    fluxes = []
    from scipy.integrate import simps
    for i in filters:
        #if FWHM is given, compute wl min and max
        if FWHM:
            band = [i[0]-i[1]/2., i[0]+i[1]/2.]
        else:
            band = i
        #print(band)
        good = np.logical_and(wl > band[0], wl < band[1]) #where band[0] is cuton, band[1] cutoff
        flux = simps(F_lam[good], wl[good])
        fluxes += [flux/(band[1]-band[0])]
        
    F_lam_in = np.array(fluxes) 
    if FWHM:
        mean_wl = filters[:,0]*1e-4
    else:
        mean_wl = np.mean(filters, axis = 1)*1e-4 #convert to cm
    
    #convert back to F_nu
    F_nu_in = mean_wl**2*F_lam_in/c
    return F_nu_in
    
###Now define functions to fit. 

def SED_to_fit(wl, Ts, Ms, f_Sis, distance):
    """Deal with multi component fit
    Ts, Ms, and f_Si are arrays of same length specify different 
    SED components.
    distance is in cm. 
    """
    dust_models = []
    if len(Ts) == len(Ms) == len(f_Sis):
        for ind in range(len(Ts)):
            T = Ts[ind]
            M = Ms[ind]
            f_Si = f_Sis[ind]
            Si_mass = f_Si*M
            C_mass =  (1-f_Si)*M
            Si_dust = dust_flux(wl, T, Si_mass, 'Si',0.1, distance = distance)
            C_dust = dust_flux(wl, T, C_mass, 'C',0.1, distance  = distance)
            dust_models += [Si_dust + C_dust]
    else:
        print("Length of Ts, Ms, and f_Si must be equal")
    dust_models = np.array(dust_models)
    total_dust_SED = np.sum(dust_models, axis = 0)
    return 1e26*total_dust_SED 


###############FIT Taking optical depth into account. Based on Shahbandeh+2023, but using emcee#

def p_esc(tau):
    "eq 3 in the paper"
    p = 3/(4*tau) * (1-1/(2*tau**2)+ (1/tau + 1/(2*tau**2))*np.exp(-2*tau) )
    for i in range(len(p)):
        if np.isnan(p[i]): 
            p[i] = 0
    return p

def tau(wl, t, Mdust_tot, v_ej, comp = 'C'):
    """
    The optical depth at different wavelength as a fuction of time,
    assuming homologous expansion.

    ALL INPUT MUST BE WITH ASTROPY UNIT
    """
    opt_depth = 3/4 * (Mdust_tot/(np.pi * v_ej**2)) * draine_kappa(wl, comp)*(u.cm**2/u.g) * t**(-2)
    return opt_depth.to(1).value

def SED_to_fit_opt_depth(wl, Ts, Ms, f_Sis, epoch, v_ej, distance):
    """Deal with multi component fit
    Ts, Ms, and f_Si are arrays of same length specify different 
    SED components.
    distance is in cm. 
    T in K
    M in solar mass
    """
    dust_models = []
    if len(Ts) == len(Ms) == len(f_Sis):
        for ind in range(len(Ts)):
            T = Ts[ind]
            M = Ms[ind]
            f_Si = f_Sis[ind]
            Si_mass = f_Si*M
            C_mass =  (1-f_Si)*M
            #Compute the optical depth
            print(Si_mass, C_mass)
            tau_Si = tau(wl*u.micron, epoch*u.day, Si_mass*u.Msun, v_ej*u.km/u.s, comp = 'Si')
            tau_C =  tau(wl*u.micron, epoch*u.day, C_mass*u.Msun , v_ej*u.km/u.s, comp = 'C')
            # print(tau_Si, tau_C)
            #Compute the escape fraction
            p_esc_Si = p_esc(tau_Si)
            p_esc_C  = p_esc(tau_C)
            print(p_esc_Si, p_esc_C)
            #Now compute the observed flux
            Si_dust = dust_flux(wl, T, Si_mass*p_esc_Si, 'Si',0.1, distance = distance)
            C_dust =  dust_flux(wl, T,  C_mass*p_esc_C , 'C',0.1, distance  = distance)
            dust_models += [Si_dust + C_dust]
    else:
        print("Length of Ts, Ms, and f_Si must be equal")
    dust_models = np.array(dust_models)
    total_dust_SED = np.sum(dust_models, axis = 0)
    return 1e26*total_dust_SED 






    