import numpy as np

def get_mom_maps(spec_cube, mask, vaxis, mom_calc =[3, 3,"fwhm"]):
    """
    Function to compute moment maps
    """

    # --- strip units ONCE ---
    spec_vals = spec_cube.value            # (n_pts, n_chan)
    v_vals    = vaxis.value                # (n_chan,)
    dv        = abs(v_vals[0] - v_vals[1])      # scalar
    spec_unit = spec_cube.unit
    v_unit    = vaxis.unit

    # --- set up output maps WITH units ---
    mom_maps = {}

    dim_sz = np.shape(spec_vals)
    n_pts  = dim_sz[0]
    # n_chan = dim_sz[1]

    SNthresh         = mom_calc[0]
    conseq_channels  = int(np.nanmax((float(mom_calc[1]),3)))
    mom2_method      = mom_calc[2]
    fac_mom2         = np.sqrt(8*np.log(2)) if mom2_method == "fwhm" else 1.0

    # Output array templates
    mom_maps["rms"]       = np.full(n_pts, np.nan) * spec_unit
    mom_maps["tpeak"]     = np.full(n_pts, np.nan) * spec_unit

    mom_maps["mom0"]      = np.full(n_pts, np.nan) * spec_unit * v_unit
    mom_maps["mom0_err"]  = np.full(n_pts, np.nan) * spec_unit * v_unit

    mom_maps["mom1"]      = np.full(n_pts, np.nan) * v_unit
    mom_maps["mom1_err"]  = np.full(n_pts, np.nan) * v_unit

    mom2_unit = v_unit if mom2_method == "fwhm" else v_unit**2
    mom_maps["mom2"]      = np.full(n_pts, np.nan) * mom2_unit
    mom_maps["mom2_err"]  = np.full(n_pts, np.nan) * mom2_unit

    mom_maps["ew"]        = np.full(n_pts, np.nan) * v_unit
    mom_maps["ew_err"]    = np.full(n_pts, np.nan) * v_unit

    # -------------------------------
    #       MAIN LOOP (NO UNITS)
    # -------------------------------
    for m in range(n_pts):

        spectrum = spec_vals[m, :]      # 1D float array
        mask_m   = mask[m, :]

        # Skip empty spectra
        if np.nansum(spectrum != 0) < 1:
            continue

        # ---------------- RMS ----------------
        rms = np.nanstd(spectrum[np.logical_and(mask_m == 0, spectrum != 0)])
        mom_maps["rms"][m] = rms * spec_unit

        # ---------------- Tpeak ----------------
        tpeak = np.nanmax(spectrum * mask_m)
        mom_maps["tpeak"][m] = tpeak * spec_unit

        # ---------------- Mom0 ----------------
        mom0 = np.nansum(spectrum * mask_m) * dv
        mom_maps["mom0"][m] = mom0 * spec_unit * v_unit

        mom0_err = np.sqrt(np.nansum(mask_m)) * rms * dv
        mom_maps["mom0_err"][m] = mom0_err * spec_unit * v_unit

        # ---------------- Build high-signal mask ----------------
        masked = (spectrum * mask_m > SNthresh * rms).astype(int)
        masked = ((masked + np.roll(masked,1) + np.roll(masked,-1)) >= 3).astype(int)

        if np.nansum(masked) < conseq_channels - 2:
            continue

        for _ in range(5):
            masked = ((masked + np.roll(masked,1) + np.roll(masked,-1)) >= 1).astype(int)

        # ---------------- Mom1 ----------------
        num1 = np.nansum(spectrum * v_vals * masked)
        den1 = np.nansum(spectrum * masked)

        mom1 = num1 / den1
        mom_maps["mom1"][m] = mom1 * v_unit

        numer = rms**2 * np.nansum(masked * (v_vals - mom1)**2)
        mom1_err = np.sqrt(numer / den1**2)
        mom_maps["mom1_err"][m] = mom1_err * v_unit

        # ---------------- Mom2 ----------------
        mom2_math = np.nansum(spectrum * masked * (v_vals - mom1)**2) / den1

        numer = rms**2 * np.nansum((masked * (v_vals - mom1)**2 - mom2_math)**2)
        mom2_err = np.sqrt(numer / den1**2)

        if mom2_method == "fwhm":
            mom_maps["mom2"][m]     = fac_mom2 * np.sqrt(mom2_math) * v_unit
            mom_maps["mom2_err"][m] = fac_mom2 * mom2_err / (2 * np.sqrt(mom2_math)) * v_unit
        else:
            mom_maps["mom2"][m]     = mom2_math * v_unit**2
            mom_maps["mom2_err"][m] = mom2_err * v_unit**2

        # ---------------- EW ----------------
        ew = np.nansum(spectrum * masked) * dv / tpeak / np.sqrt(2*np.pi)
        mom_maps["ew"][m] = ew * v_unit

        term1 = rms**2 * np.nansum(masked) * dv**2 / (2*np.pi * tpeak**2)
        term2 = (ew**2 - ew * dv/np.sqrt(2*np.pi))
        ew_err = np.sqrt(term1 + term2)
        mom_maps["ew_err"][m] = ew_err * v_unit

    return mom_maps