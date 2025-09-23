import numpy as np
import pandas as pd
from scipy import stats
from astropy.stats import median_absolute_deviation, mad_std
from mom_computer import get_mom_maps

from structure_addition import *
from shuffle_spec import *


def construct_mask(ref_line, this_data, SN_processing):
    """
    Function to construct the mask based on high and low SN cut
    """
    ref_line_data = this_data["SPEC_VAL_"+ref_line]
    n_pts = np.shape(ref_line_data)[0]
    n_chan = np.shape(ref_line_data)[1]

    line_vaxis = this_data['SPEC_VCHAN0']+(np.arange(n_chan)-(this_data['SPEC_CRPIX']-1))*this_data['SPEC_DELTAV']

    line_vaxis = line_vaxis/1000 #to km/s
    #Estimate rms
    rms = median_absolute_deviation(ref_line_data, axis = None, ignore_nan = True)
    rms = median_absolute_deviation(ref_line_data[np.where(ref_line_data<3*rms)], ignore_nan = True)

    # Mask each spectrum
    low_tresh, high_tresh = SN_processing[0], SN_processing[1]
    mask = np.array(ref_line_data > high_tresh * rms, dtype = int)
    low_mask = np.array(ref_line_data > low_tresh * rms, dtype = int)

    mask = mask & (np.roll(mask, 1,1) | np.roll(mask,-1,1))

    #remove spikes along spectral axis:
    mask = np.array((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1))>=3, dtype = int)
    low_mask = np.array((low_mask + np.roll(low_mask, 1, 1) + np.roll(low_mask, -1, 1))>=3, dtype = int)

    #remove spikes along spatial axis:
    #mask = np.array((mask + np.roll(mask, 1, 0) + np.roll(mask, -1, 0))>=3, dtype = int)
    #low_mask = np.array((low_mask + np.roll(mask, 1, 0) + np.roll(low_mask, -1, 0))>=3, dtype = int)

    #expand to cover all > 2sigma that have a 2-at-4sigma core
    for kk in range(5):
        mask = np.array(((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) >= 1), dtype = int)*low_mask

    #expand to cover part of edge of the emission line
    for kk in range(2):
        mask = np.array(((mask + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) >= 1), dtype = int)
    
    # Derive the ref line mean velocity
    line_vmean = np.zeros(n_pts)*np.nan

    for jj in range(n_pts):
        line_vmean[jj] = np.nansum(line_vaxis * ref_line_data[jj,:]*mask[jj,:])/ \
                       np.nansum(ref_line_data[jj,:]*mask[jj,:])

    return mask, line_vmean, line_vaxis

def dist(ra, dec, ra_c, dec_c):
    return np.sqrt((ra-ra_c)**2+(dec-dec_c)**2)
    
def process_spectra(sources_data,  # LN: variable not used
                    source_list,
                    lines_data,
                    fname,shuff_axis,
                    run_success,
                    ref_line_method,
                    SN_processing,
                    strict_mask,
                    input_mask = None, # newly added
                    use_input_mask = False,  # newly added
                    mom_calc = [3, 3, "fwhm"],
                    just_source = None
                    ):
    """
    :param sources_data: Pandas DataFrame which is the geometry.txt file
    :param lines_data:   Pandas DataFrame which is the cubes_list.txt
    """
    
    n_sources = len(source_list)
    n_lines = len(lines_data["line_name"])
    if ref_line_method in list(lines_data["line_name"]):
        #user defined reference line
        ref_line = ref_line_method.upper()
    else:
        ref_line = lines_data["line_name"][0].upper()

    for ii in range(n_sources):


        #if the run was not succefull, don't do processing of the data
        if not run_success[ii]:
            continue

        this_source = source_list[ii]
        if not just_source is None:
            if just_source != this_source:
                continue



        print("----------------------------------")
        print(f'Source: {this_source}')
        print("----------------------------------")

        this_data = np.load(fname[ii],allow_pickle = True).item()
        tags = this_data.keys()
        n_chan = np.shape( this_data["SPEC_VAL_"+ref_line])[1]
        #--------------------------------------------------------------
        #  Build a mask based on reference line(s)
        #--------------------------------------------------------------

        if use_input_mask:
            # check that input mask is provided
            if len(input_mask) == 0:
                print(f'{"[ERROR]":<10}', f'No mask provided!')

            # use input mask
            # mask = this_data["SPEC_VAL_MASK"]
            mask = this_data[f'SPEC_VAL_{input_mask["mask_name"][0].upper()}']

            # clean up database
            del this_data[f'SPEC_VAL_{input_mask["mask_name"][0].upper()}']
            del this_data[f'SPEC_DESC_{input_mask["mask_name"][0].upper()}']
            # del this_data["SPEC_DESC_MASK"]

            # take reference velocity and vaxis from reference line
            _, ref_line_vmean, ref_line_vaxis = construct_mask(ref_line, this_data, SN_processing)

        else:
            # Use function for mask
            mask, ref_line_vmean, ref_line_vaxis = construct_mask(ref_line, this_data, SN_processing)
            this_data["SPEC_MASK_"+ref_line]= mask
            #this_data["INT_VAL_V"+ref_line] = ref_line_vmean

            #check if all lines used as reference line
            n_mask = 0
            if ref_line_method in ["all"]:
                n_mask = n_lines
                print(f'{"[INFO]":<10}', 'All lines used as prior.')
            elif isinstance(ref_line_method, int):
                n_mask = np.min([n_lines,ref_line_method])
                print(f'{"[INFO]":<10}', f'Using first {n_mask+1} lines as prior.')
            if n_mask>0:
                for n_mask_i in range(1,n_mask+1):
                    line_i = lines_data["line_name"][n_mask_i].upper()
                    mask_i, ref_line_vmean_i, ref_line_vaxis_i = construct_mask(line_i, this_data, SN_processing)
                    this_data["SPEC_MASK_"+line_i]= mask_i
                    #this_data["INT_VAL_V"+line_i] = ref_line_vmean_i

                    # add mask to existing mask
                    mask = mask | mask_i
                    
            elif ref_line_method in ["ref+HI"]:
                if "hi" not in list(lines_data["line_name"]):
                    print(f'{"[WARNING]":<10}', 'HI not in PyStructure. Skipping.')
                else:
                    mask_hi, ref_line_vmean_hi, ref_line_vaxis_hi = construct_mask("HI", this_data, SN_processing)
                    
                    mask = mask | mask_hi
                    
                    rgal = this_data["rgal_r25"]
                    n_pts = len(this_data["rgal_r25"])
                    vmean_comb = np.zeros(n_pts)*np.nan
                    for jj in range(n_pts):
                        if rgal[jj]<0.23:
                            vmean_comb[jj] = ref_line_vmean[jj]
                        else:
                            vmean_comb[jj] = ref_line_vmean_hi[jj]
                    ref_line_vmean = vmean_comb
            if strict_mask:
                """
                Make sure that spatialy we do not have only connected pixels
                """
                ra, dec = this_data["ra_deg"], this_data["dec_deg"]
                for jj in range(n_chan):
                    mask_spec = mask[:,jj]
                    mask_labels=np.zeros_like(mask_spec)
                    sep = this_data["beam_as"]/3600/2
                    label=1
                    for n in range(len(mask_labels)):
                        if mask_labels[n]==0:
                            if mask_spec[n]==0:
                                mask_labels[n]=-99
                                continue
            
                            dist_array=dist(ra, dec, ra[n], dec[n])
                            #check out neighbours
                            idx_neigh=np.where(abs(dist_array-sep)<0.1*this_data["beam_as"]/3600)
                            #check if labels have already been given (except 0 or -99)
                            labels_given=np.unique(mask_labels[idx_neigh])
                            index = labels_given[labels_given>0]
                            if len(index)>0:
                                mask_labels[n]=index[0]
                                if len(index)>1:
                                    for i in range(len(index)-1):
                                        mask_labels[mask_labels==index[i+1]]=index[0]
                            else:
                                mask_labels[n]=label
                                label+=1
                    labels = np.unique(mask_labels)
                    for lab in labels:
                        if lab <0:
                            continue
                        if len(mask[:,jj][np.where(mask_labels==lab)])<5:
                            mask[:,jj][np.where(mask_labels==lab)]=0

        #store the mask in the PyStructure
        this_data["SPEC_MASK"]= mask
        this_data["INT_VAL_VSHUFF"] = ref_line_vmean

        #-------------------------------------------------------------------
        # Apply the CO-based mask to the EMPIRE lines and shuffle them
        #-------------------------------------------------------------------
        n_chan_new = 200
       
        for jj in range(n_lines):
            line_name = lines_data["line_name"][jj].upper()

            # need to add band structure, if the 2D was not yet provided
            if lines_data["band_ext"].isnull()[jj]:
                this_data = add_band_to_struct(struct = this_data, \
                                           band = line_name,\
                                           unit = 'K km/s', \
                                           desc = line_name + ' Shuffled by '+ref_line)

            this_data = add_spec_to_struct(struct= this_data, \
                                           line = "SHUFF"+line_name,\
                                           unit = "K",\
                                           desc = line_name + ' Shuffled by '+ref_line,\
                                           n_chan = n_chan_new)


            if not 'SPEC_VAL_'+line_name in this_data.keys():
                print(f'{"[ERROR]":<10}', f'Tag for line {line_name} not found. Proceeding.')
                continue
            this_spec = this_data['SPEC_VAL_'+line_name]
            if np.nansum(this_spec, axis = None)==0:
                print(f'{"[ERROR]":<10}', f'Line {line_name} appears empty. Skipping.')
                continue

            dim_sz = np.shape(this_spec)
            n_pts = dim_sz[0]
            n_chan = dim_sz[1]
            this_v0 = this_data["SPEC_VCHAN0"]
            this_deltav = this_data["SPEC_DELTAV"]
            this_crpix = this_data["SPEC_CRPIX"]
            
            this_vaxis = (this_v0 + (np.arange(n_chan)-(this_crpix-1))*this_deltav)/1000 #to km/s
            this_data["SPEC_VAXIS"] = this_vaxis

            shuffled_mask = shuffle(spec = mask, \
                                    vaxis = ref_line_vaxis,\
                                    zero = 0.0,\
                                    new_vaxis = this_vaxis, \
                                    interp = 0)
                        
            #compute moment_maps
            mom_maps = get_mom_maps(this_spec, shuffled_mask,this_vaxis, mom_calc)

            # Save in structure
            if lines_data["band_ext"].isnull()[jj]:

                tag_ii = "INT_VAL_"+line_name
                tag_uc = "INT_UC_" + line_name
                
                tag_tpeak = "INT_TPEAK_" + line_name
                tag_rms = "INT_RMS_" + line_name
                
                tag_mom1 = "INT_MOM1_" + line_name
                tag_mom1_err = "INT_EMOM1_" + line_name
                
                #Note that Mom2 corresponds to a FWHM
                tag_mom2 = "INT_MOM2_" + line_name
                tag_mom2_err = "INT_EMOM2_" + line_name
                
                tag_ew = "INT_EW_" + line_name
                tag_ew_err = "INT_EEW_" + line_name
                
                # store the different calculations
                this_data[tag_ii] = mom_maps["mom0"]
                this_data[tag_uc] = mom_maps["mom0_err"]
                this_data[tag_tpeak] = mom_maps["tpeak"]
                this_data[tag_rms] = mom_maps["rms"]
                this_data[tag_mom1] = mom_maps["mom1"]
                this_data[tag_mom1_err] = mom_maps["mom1_err"]
                this_data[tag_mom2] = mom_maps["mom2"]
                this_data[tag_mom2_err] = mom_maps["mom2_err"]
                
                this_data[tag_ew] = mom_maps["ew"]
                this_data[tag_ew_err] = mom_maps["ew_err"]
                
                #-------------------------------------------------
                #!!!!!!!! Will be depricated in future update!!!!!
                #tag_tpeak_dep = "SPEC_TPEAK_" + line_name
                #tag_rms_dep = "SPEC_RMS_" + line_name
                #this_data[tag_tpeak_dep] = mom_maps["tpeak"]
                #this_data[tag_rms_dep] = mom_maps["rms"]
                #-------------------------------------------------
            else:
                print(f'{"[INFO]":<10}', f'Intensity Map for {lines_data["line_name"][jj]} already provided. Skipping.')

            #Shuffle the line
            #;- DC modify 02 march 2017: define a reference velocity axis
            #;-   this_deltav varies from dataset to dataset (fixing bug for inverted CO21 vaxis)
            cdelt = shuff_axis[1]
            naxis_shuff = int(shuff_axis[0])
            new_vaxis = cdelt * (np.arange(naxis_shuff)-naxis_shuff/2)
            new_vaxis=new_vaxis/1000 #to km/s

            shuffled_line = shuffle(spec = this_spec,\
                                    vaxis = this_vaxis,\
                                    zero = ref_line_vmean,
                                    new_vaxis = new_vaxis,\
                                    interp = 0)

            tag_i = "SPEC_VAL_SHUFF" + line_name
            tag_v0 = "SPEC_VCHAN0_SHUFF"
            tag_deltav = "SPEC_DELTAV_SHUFF"


            this_data[tag_i] = shuffled_line
            this_data[tag_v0] = new_vaxis[0]
            this_data[tag_deltav] = (new_vaxis[1] - new_vaxis[0])

            this_data["SPEC_VAXISSHUFF"] = new_vaxis
        
        this_data["SPEC_CRPIX_SHUFF"] = 1
        np.save(fname[ii], this_data)


        # /__
