import numpy as np
import pandas as pd
from scipy import stats
from astropy.stats import median_absolute_deviation, mad_std
from astropy import units as au


def get_mom_maps(spec_cube, mask, vaxis, mom_calc =[3, 3,"fwhm"]):
    """
    Function to compute moment maps
    """
    #define the moment map
    mom_maps = {}

    dim_sz = np.shape(spec_cube)
    n_pts = dim_sz[0]
    n_chan = dim_sz[1]
    delta_v = abs(vaxis[0]-vaxis[1])

    SNthresh = mom_calc[0]
    conseq_channels = int(np.nanmax((float(mom_calc[1]),3)))
    mom2_method = mom_calc[2]
    #check if we do fwhm or not for mom2
    fac_mom2 = 1
    if mom2_method in ["fwhm"]:
        #factor to convert sqrt(mom2) into fwhm
        fac_mom2 = np.sqrt(8*np.log(2))
    #first we iterate over the individual spectra:

    mom_maps["rms"] = np.zeros(n_pts)*np.nan * spec_cube.unit
    mom_maps["tpeak"] = np.zeros(n_pts)*np.nan * spec_cube.unit

    mom_maps["mom0"]= np.zeros(n_pts)*np.nan * spec_cube.unit * vaxis.unit
    mom_maps["mom0_err"]= np.zeros(n_pts)*np.nan * spec_cube.unit *vaxis.unit

    mom_maps["mom1"]= np.zeros(n_pts)*np.nan * vaxis.unit
    mom_maps["mom1_err"]= np.zeros(n_pts)*np.nan * vaxis.unit

    #Note: We will convert the mathematical mom2 term to FWHM
    if fac_mom2 in ["math"]:
        unit_mom2 =vaxis.unit**2
    else:
        unit_mom2 =vaxis.unit

    mom_maps["mom2"] = np.zeros(n_pts)*np.nan * unit_mom2
    mom_maps["mom2_err"]= np.zeros(n_pts)*np.nan * unit_mom2

    mom_maps["ew"] = np.zeros(n_pts)*np.nan * vaxis.unit
    mom_maps["ew_err"]= np.zeros(n_pts)*np.nan * vaxis.unit



    for m in range(n_pts):
        #skip if spectrum is empty
        if np.nansum(spec_cube[m,:]!=0, axis = None)>=1:
            #compute rms of spectrum
            mom_maps["rms"][m] = np.nanstd(\
            spec_cube[m,:][np.where(np.logical_and(\
                mask[m,:]==0, spec_cube[m,:]!=0))]) * spec_cube.unit
            mom_maps["tpeak"][m] = np.nanmax(spec_cube[m,:]*mask[m,:], axis = 0)

            
            mom_maps["mom0"][m] = np.nansum(spec_cube[m,:]*mask[m,:], axis = 0)*delta_v
            mom_maps["mom0_err"][m] = np.sqrt(np.nansum(mask[m,:]))*mom_maps["rms"][m]*delta_v

            #compute line_specific mask
            masked = np.array(spec_cube[m,:]*mask[m,:]>SNthresh*mom_maps["rms"][m], dtype = int)
            masked = np.array((masked + np.roll(masked, 1) + np.roll(masked, -1))>=3, dtype = int)

            if np.nansum(masked)>=conseq_channels-2:
                for kk in range(5):
                    masked = np.array(((masked + np.roll(masked, 1) + np.roll(masked, -1)) >= 1), dtype = int)*au.dimensionless_unscaled

                #compute higher moments only, if enough high signal spaxels:
                #mom1----------------------------------------------------------------
                mom_maps["mom1"][m] = np.nansum(spec_cube[m,:]*vaxis*masked) / np.nansum(spec_cube[m,:]*masked)

                sum_T = np.nansum(spec_cube[m,:]*masked)
                numer = mom_maps["rms"][m]**2*np.nansum(masked*(vaxis-mom_maps["mom1"][m])**2)
                mom_maps["mom1_err"][m] = (numer/sum_T**2)**0.5

                #mom2----------------------------------------------------------------
                #Note: We convert the mathematical mom2 term to FWHM
                mom2 = np.nansum(spec_cube[m,:]*masked*(vaxis-mom_maps["mom1"][m])**2) / np.nansum(spec_cube[m,:]*masked)

                
                numer = mom_maps["rms"][m]**2*np.nansum((masked*(vaxis-mom_maps["mom1"][m])**2-mom2)**2)
                mom2_err = (numer/sum_T**2)**0.5

                #default: convert mom2 to FWHM (km/s)
                if fac_mom2 in ["math"]:
                    #use mathematical definition (units are ([L]/[T])^2)
                    mom_maps["mom2"][m] = mom2
                    mom_maps["mom2_err"][m] = mom2_err
                else:
                    mom_maps["mom2"][m] = fac_mom2*np.sqrt(mom2)
                    
                    mom_maps["mom2_err"][m] = fac_mom2*mom2_err/np.sqrt(mom2)/2

                #ew--------------------------------------------------------------------
                mom_maps["ew"][m] = np.nansum(spec_cube[m,:]*masked)*delta_v / np.nanmax(spec_cube[m,:]*masked) / np.sqrt(2 * np.pi)
                
                #JdB: Error due to improper units, temporary fix: comment out last part
                term1= mom_maps["rms"][m]**2*np.nansum(masked)*delta_v**2 / (2*np.pi*np.nanmax(spec_cube[m,:]*masked)**2)
                term2= (mom_maps["ew"][m]**2 - mom_maps["ew"][m]*delta_v/np.sqrt(2*np.pi))#*mom_maps["rms"][m]**2
                mom_maps["ew_err"][m] = (term1 + term2)**0.5

    return mom_maps
