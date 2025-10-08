"""
This routine generates a database, similar to the struct in idl.
Several side functions are part of this routine.
The original routine was wrirtten in idl (see create_database.pro)

The output is an astropy table saved as an .ecsv file. To open it in a new 
python script use for example:
#### TBD: CHANGE TO ASTROPY TABLE
    > read_dictionary = np.load('datafile.npy',allow_pickle = True).item()
    > ra_samp = read_dictionary["ra_deg"]
    > dec_samp = read_dictionary["dec_deg"]
    > intensity = np.nansum(read_dictionary["SPEC_VAL_CO21"], axis = 0)
    > plt.scatter(ra_samp, dec_samp, c = intensity, marker = "h")

MODIFICATION HISTORY

    - v1.0.1 16-22 October 2019: Conversion from IDL to Python
        Minor changes implemented
        ToDo:
            - now can only read in (z,x,y) cubes, but should be flexible to
              recognize (1,z,x,y) cubes as well

    - v1.1.1 26 October 2020: More stable version. Several bugs fixed.
            - Used by whole Bonn group

    - v1.2.1 January 2022
            - Implemented customization of reference line for masking.
              Now several lines can be defined for the mask creation

    - v1.2.2 January 2022
            - Implement Moment 1, Moment 2 and EW calculation
            - Restructured INT and SPEC keys (Mom maps now in INT keys)

    - v2.0.0 January 2022
            - Implemented config file: You can run the PyStructure using a single config file

    - v2.0.1. January 2022
            - Automatically determine the max radius for the sampling points

    - v2.1.0. July 2022
            - Include Spectral Smooting and Convolving for data with significantly different spectral resolution.

    - v2.1.1. October 2022
            - Save moment maps as fits file

    - v3.0.0. August 2023
            - Clean up: Remove unnecessary keys
            - Improve masking -> Remove spurious spatial spikes

    - v3.0.1 January 2024
            - Fix error map convolution handeling

    - v3.1.0 July 2025
            - Merge publishes version with Uni Bonn version
            - Implement feature to complete PyStructre

    - v3.1.1 September 2025
            - Input velocity-integration mask as optional feature
            - Clean-ups to improve readibility of the code

    - v4.0.1 October 2025
            - Major change: Change the infrastructure from numpy dictonary to Astropy Tables

    - v4.1.0 October 2025
            - New (optional) feature: Masking of fine structure lines

"""
__author__ = "J. den Brok & L. Neumann"
__version__ = "v4.1.0"
__email__ = "jadenbrok@mpia.de & lukas.neumann@eso.org"
__credits__ = ["M. Jimenez-Donaire", "E. Rosolowsky", "A. Leroy ", "I. Beslic"]


import numpy as np
import pandas as pd
import os.path
from os import path
import shutil
from astropy.io import fits
from datetime import date, datetime
import re
import argparse
from astropy.table import Table, Column
from astropy import units as au
today = date.today()
date_str = today.strftime("%Y_%m_%d")
import glob

import sys
sys.path.append("./scripts/")
from structure_addition import add_band_to_struct, add_spec_to_struct
from sampling import make_sampling_points
from sampling_at_resol import sample_at_res, sample_mask
from deproject import deproject
from twod_header import twod_head
from processing_spec import process_spectra
from message_list import *
from save_moment_maps import save_mom_to_fits




#----------------------------------------------------------------------
# The function that generates an empty directory
#----------------------------------------------------------------------

def fill_checker(fname, sample_coord, bands, cubes):
    """
    Function that checks if a given PyStructure exists and matches in terms of the sampling points.
    :param fname: PyStructure Filename
    :param sample_coord: [samp_ra, samp_dec] - The sample coordinates. Used to match to existing PySturcture.
    :param bands:
    :param cubes
    """

    this_data = Table.read(fname)

    #Check 1: Enusre that the coordinates are identical (1e-12 to allow wiggle room due to rounding errors)
    if abs(np.nansum(this_data['ra_deg']-sample_coord[0]*au.deg)) + abs(np.nansum(this_data['dec_deg']-sample_coord[1]*au.deg))>1e-12*au.deg:
        raise ValueError('The PyStructure does not match. Please run code setting the "structure_creation" key to "overwrite"')
    
    #Check 2: Now check which bands and cubes 
    fill_bands = []
    for band_nm in bands["band_name"]:
        if f'BAND_{band_nm.upper()}' in  list(this_data.keys()):
            fill_bands.append(band_nm)
    fill_cubes = []
    for cube_nm in cubes["line_name"]:
        if f'MOM0_{cube_nm.upper()}' in  list(this_data.keys()):
            fill_cubes.append(cube_nm)
    return this_data, fill_bands,fill_cubes

def create_temps(conf_file):
    """
    Separeate the config file into variables, band and cube list
    """
    loc = 0
    py_input ='./Temp_Files/conf_Py.py'
    band_f = './Temp_Files/band_list_temp.txt'
    cube_f = './Temp_Files/cube_list_temp.txt'
    mask_f = './Temp_Files/mask_temp.txt'

    with open(conf_file,'r') as firstfile, open(py_input,'a') as secondfile, open(band_f,'a') as third, open(cube_f,'a') as fourth, open(mask_f,'a') as fifth:

        # read content from first file
        for line in firstfile:
            # append content to second file
            if "Define Bands" in line:
                loc = 1
            if "Define Cubes" in line:
                loc = 2            
            if "Define Mask" in line:
                loc = 3

            if loc == 0:
                secondfile.write(line)
            elif loc == 1:
                third.write(line)
            elif loc == 2:
                fourth.write(line)
            elif loc == 3:
                fifth.write(line)

    return band_f, cube_f, mask_f

def create_database(just_source=None, quiet=False, conf=False):
    """
    Function that generates a python dictionary containing a hexagonal grid.
    :param just_source: String name of a source, if one wants only one source
    :param quiet: Verbosity set to mute
    :param conf: Config File provided
    :return database: python dictionary
    """

    if quiet == False:
        print(f'{"[INFO]":<10}', 'Reading in source parameters.')
    names_source = ["source", "ra_ctr", "dec_ctr", "dist_mpc", "e_dist_mpc",
                  "incl_deg", "e_incl_deg","posang_deg", "e_posang_deg",
                  "r25", "e_r25"]
    source_data = pd.read_csv(geom_file, sep = "\t",names = names_source,
                            comment = "#")

    #define list of sources (need to differentiate between conf file input and default)
    if conf:
        if isinstance(sources, tuple):
            source_list = list(sources)
        else:
            source_list = [sources]

    else:
        source_list = list(source_data["source"])

    n_sources = len(source_list)
    # -----------------------------------------------------------------
    # GENERATE THE EMPTY DATA STRUCTURE
    # -----------------------------------------------------------------

    # Add the bands to the structure
    band_columns = ["band_name","band_desc", "band_unit",
                    "band_ext", "band_dir","band_uc" ]
    bands = pd.read_csv(band_file, names = band_columns, sep='[\s,]{2,20}', comment="#")

    n_bands = len(bands["band_name"])
    
    if quiet == False:
        print(f'{"[INFO]":<10}', f'{n_bands} band(s) loaded into structure.')


    # Add the cubes to the structure
    cube_columns = ["line_name", "line_desc", "line_unit", "line_ext", "line_dir" , "band_ext", "band_uc"]

    cubes = pd.read_csv(cube_file, names = cube_columns, sep='[\s,]{2,20}', comment="#")
    n_cubes = len(cubes["line_name"])


    if quiet == False:
        print(f'{"[INFO]":<10}', f'{n_cubes} cube(s) loaded into structure.')

    # Add the input velocity-integration mask to the structure
    # mask_columns = ["mask_name", "mask_desc", "mask_unit", "mask_ext", "mask_dir"]
    mask_columns = ["mask_name", "mask_desc", "mask_ext", "mask_dir"]
    input_mask = pd.read_csv(mask_file, names = mask_columns, sep='[\s,]{2,20}', comment="#")
    if len(input_mask) == 0:
        if quiet == False:
            print(f'{"[INFO]":<10}', f'No mask provided; will be constructed from prior line(s).')
    else:
        if quiet == False:
            if use_input_mask:
                print(f'{"[INFO]":<10}', f'Input mask loaded into structure; will be used for products.')
            else:
                print(f'{"[INFO]":<10}', f'Input mask loaded into structure; will NOT be used for products.')


    #-----------------------------------------------------------------
    # LOOP OVER SOURCES
    #-----------------------------------------------------------------

    #additional parameters
    run_success = [True]*n_sources #keep track if run succesfull for each source
    fnames=[""]*n_sources   #filename save for source
    overlay_hdr_list = []
    overlay_slice_list = []

    for ii in range(n_sources):
        #if config file provided, use the list of sources provided therein

        this_source = source_list[ii]

        if not this_source in list(source_data["source"]):
            run_success[ii]=False

            print(f'{"[ERROR]":<10}', f'{this_source} not in source table.')

            continue

        #assign correct index of list and input source (relevant for index file)
        ii_list = np.where(np.array(source_data["source"])==this_source)[0][0]


        if not just_source is None:
            if this_source != just_source:
                continue

        print("-------------------------------")
        print(f'Source: {this_source}')
        print("-------------------------------")

        #---------------------------------------------------------------------
        # MAKE SAMPLING POINTS FOR THIS TARGET
        #---------------------------------------------------------------------

        #Generate sampling points using the overlay file provided as a template and half-beam spacing.


        # check if overlay name given with or without the source name in it:
        if this_source in overlay_file:
            overlay_fname = data_dir+overlay_file
        else:
            overlay_fname = data_dir+this_source+overlay_file


        if not path.exists(overlay_fname):
            run_success[ii]=False

            print(f'{"[ERROR]":<10}', f'No Overlay data found. Skipping {this_source}. Check path to overlay file.')
            overlay_hdr_list.append("")
            overlay_slice_list.append("")
            continue


        ov_cube,ov_hdr = fits.getdata(overlay_fname, header = True)


        #check, that cube is not 4D
        if ov_hdr["NAXIS"]==4:
            run_success[ii]=False
            overlay_hdr_list.append("")
            overlay_slice_list.append("")
            print(f'{"[ERROR]":<10}', f'4D cube provided. Need 3D overlay. Skipping {this_source}.')
            continue

        #add slice of overlay
        overlay_slice_list.append(ov_cube[ov_hdr["NAXIS3"]//2,:,:])

        mask = np.sum(np.isfinite(ov_cube), axis = 0)>=1
        mask_hdr = twod_head(ov_hdr)
        overlay_hdr_list.append(mask_hdr)
        if resolution == 'native':
            target_res_as = np.max([ov_hdr['BMIN'], ov_hdr['BMAJ']]) * 3600
        elif resolution == 'physical':
            target_res_as = 3600 * 180/np.pi * 1e-6 * target_res / source_data['dist_mpc'][ii_list]
        elif resolution == 'angular':
            target_res_as = target_res
        else:
            print(f'{"[ERROR]":<10}', 'Resolution keyword has to be "native", "angular" or "physical".')


        # Save the database
        if resolution == 'native':
            res_suffix = str(target_res_as).split('.')[0]+'.'+str(target_res_as).split('.')[1][0]+'as'
        elif resolution == 'angular':
            res_suffix = str(target_res_as).split('.')[0]+'as'
        elif resolution == 'physical':
            res_suffix = str(target_res).split('.')[0]+'pc'

        #Define filename used to store the PyStructure
        fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+'.ecsv'

        if "archive" in structure_creation:
            #check if basic file already exists. Otherwise, start with version numbering
            if os.path.exists(fname_dict):
                file_version=1
                fname_dict = fname_dict[:-4]+f"_v{file_version}.ecsv"
                while os.path.exists(fname_dict):
                    file_version+=1
                    fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+f'_v{file_version}.ecsv'
                if quiet == False:
                    print(f'{"[INFO]":<10}', f'Creating file version v{file_version}.')

        #check if an existing PyStructure should be completed and the user has provided a PyStructure file
        if "fill" in structure_creation:
            if 'fname_fill' in globals():
                if os.path.exists(out_dic+fname_fill):
                    fname_dict = out_dic+fname_fill
                #need to check that most recent fname_dict exists
            else:
                if not os.path.isfile(fname_dict):
        
                    print(f'{"[WARNING]":<10}', f'File {os.path.basename(fname_dict)} not found. Looking for most recent matching file...')
        
                    dir_path = os.path.dirname(fname_dict) or '.'
                    base_prefix = os.path.basename(fname_dict)[:-14]

                    possible_files = glob.glob(dir_path+"/"+base_prefix+"*")
        
        
                    if not possible_files:
                        raise FileNotFoundError(f"No file matching pattern '{base_prefix}_YYYY_MM_DD.npy' found in '{dir_path}'.")

                    # Sort by date descending and take the most recent one
        
                    fname_dict = np.sort(glob.glob(dir_path+"/"+base_prefix+"*"))[-1]
                    print(f'{"[INFO]":<10}',f"Using most recent file instead: {os.path.basename(fname_dict)}")
        
        fnames[ii] = fname_dict        
            
        
        # Determine
        spacing = target_res_as / 3600. / spacing_per_beam

        samp_ra, samp_dec = make_sampling_points(
                             ra_ctr = source_data["ra_ctr"][ii_list],
                             dec_ctr = source_data["dec_ctr"][ii_list],
                             max_rad = max_rad,
                             spacing = spacing,
                             mask = mask,
                             hdr_mask = mask_hdr,
                             overlay_in = overlay_fname,
                             show = False
                             )
        if not quiet:
            print(f'{"[INFO]":<10}', 'Finished generating hexagonal grid.')

        #---------------------------------------------------------------------
        # INITIIALIZE THE NEW STRUCTURE
        #---------------------------------------------------------------------
        n_pts = len(samp_ra)

        # The following lines do this_data=replicate(empty_struct, 1)

        if 'fill' in structure_creation:
            this_data, fill_bands, fill_cubes = fill_checker(fname_dict, [samp_ra, samp_dec], bands, cubes) 
        else:
            this_data = Table()

            #Define spectral axis of overlay cube
            #ToDo: Implement check if CUNIT3 not available
            unit_vaxis = ov_hdr["CUNIT3"]
            this_data.meta['SPEC_VCHAN0'] = ov_hdr["CRVAL3"] * au.Unit(unit_vaxis)
            this_data.meta['SPEC_DELTAV'] = ov_hdr["CDELT3"] * au.Unit(unit_vaxis)
            this_data.meta['SPEC_CRPIX'] = ov_hdr["CRPIX3"]

            # Some basic parameters for each source:
            this_data.meta["source"] = this_source
            this_data["ra_deg"] = Column(samp_ra ,unit= au.deg, description='Right ascension (J2000)')
            this_data["dec_deg"] = Column(samp_dec ,unit= au.deg, description='Declination (J2000)')
            this_data.meta["dist_mpc"] = source_data["dist_mpc"][ii_list] * au.Mpc
            this_data.meta["posang_deg"] = source_data["posang_deg"][ii_list] * au.deg
            this_data.meta["incl_deg"] = source_data["incl_deg"][ii_list] * au.deg
            this_data.meta["beam_as"] = target_res_as * au.arcsec

            # Convert to galactocentric cylindrical coordinates
            rgal_deg, theta_rad = deproject(samp_ra, samp_dec,
                                        [source_data["posang_deg"][ii_list],
                                         source_data["incl_deg"][ii_list],
                                         source_data["ra_ctr"][ii_list],
                                         source_data["dec_ctr"][ii_list]
                                        ], vector = True)


            this_data["rgal_as"] = Column(rgal_deg * 3600 , unit= au.arcsec, description='(deprojected) galactocentric radius')
            this_data["rgal_kpc"] = Column((np.deg2rad(rgal_deg)*this_data.meta["dist_mpc"]).to(au.kpc), description='(deprojected) galactocentric radius')
            this_data["rgal_r25"] = Column(rgal_deg/(source_data["r25"][ii_list]/60.),  description='(deprojected) galactocentric radius')
            this_data["theta_rad"] = Column(theta_rad , unit= au.rad, description='(deprojected) polar coordinates angle')

        #---------------------------------------------------------------------
        # LOOP OVER MAPS, CONVOLVING AND SAMPLING
        #---------------------------------------------------------------------

        for jj in range(n_bands):
            if 'fill' in structure_creation:
                if bands["band_name"][jj] in fill_bands:
                    continue
                #need to add an entry into the PyStructure

            #check if comma is in filename (in case no unc file is provided, but comma is left)
            if "," in bands["band_dir"][jj]:
                bands["band_dir"][jj] = bands["band_dir"][jj].split(',')[0]
                print(f'{"[WARNING]":<10}', f'Comma removed from band directory name for {this_source}.')

            this_band_file = bands["band_dir"][jj] + this_source + bands["band_ext"][jj]
            if not path.exists(this_band_file):
                print(f'{"[ERROR]":<10}', f'Band {bands["band_name"][jj]} not found for {this_source}.')

                continue

            if "/beam" in bands["band_unit"][jj]:
                perbeam = True
            else:
                perbeam = False
            this_int, this_hdr = sample_at_res(in_data=this_band_file,
                                     ra_samp = samp_ra,
                                     dec_samp = samp_dec,
                                     target_res_as = target_res_as,
                                     target_hdr = ov_hdr,
                                     show = False,
                                     line_name =bands["band_name"][jj],
                                     source =this_source,
                                     path_save_fits = data_dir,
                                     save_fits = save_fits,
                                     perbeam = perbeam)


            this_tag_name = 'BAND_' + bands["band_name"][jj].upper()
            this_unit = bands["band_unit"][jj]
            this_data[this_tag_name] = Column(this_int , unit= au.Unit(this_unit), description=bands["band_desc"][jj])
           
            #; MJ: ...AND ALSO THE UNCERTAINTIES FOR THE MAPS
            if  not isinstance(bands["band_uc"][jj], str):
                print(f'{"[WARNING]":<10}', f'No uncertainty band {bands["band_name"][jj]} provided for {this_source}.')
                continue
            this_uc_file = bands["band_dir"][jj] + this_source + bands["band_uc"][jj]
            if not path.exists(this_uc_file):
                print(f'{"[WARNING]":<10}', f'Uncertainty band {bands["band_name"][jj]} not found for {this_source}.')
                continue
            print(f'{"[INFO]":<10}', f'Convolving and sampling band {bands["band_name"][jj]} for {this_source}.')

            this_uc, this_hdr = sample_at_res(in_data = this_uc_file,
                                    ra_samp = samp_ra,
                                    dec_samp = samp_dec,
                                    target_res_as = target_res_as,
                                    target_hdr = ov_hdr,
                                    perbeam = perbeam,
                                    unc=True)
            this_tag_name = 'EBAND_'+bands["band_name"][jj].upper()
            this_data[this_tag_name] = Column(this_uc , unit= au.Unit(this_unit), description=f'Stats error on {bands["band_desc"][jj]}')
            

        #---------------------------------------------------------------------
        # LOOP OVER CUBES, CONVOLVING AND SAMPLING
        #---------------------------------------------------------------------

        for jj in range(n_cubes):

            if 'fill' in structure_creation:
                if cubes["line_name"][jj] in fill_cubes:
                    continue

            this_line_file = cubes["line_dir"][jj] + this_source + cubes["line_ext"][jj]


            if not path.exists(this_line_file):

                print(f'{"[ERROR]":<10}', f'Line {cubes["line_name"][jj]} not found for {this_source}.')

                continue
            print(f'{"[INFO]":<10}', f'Convolving and sampling line {cubes["line_name"][jj]} for {this_source}.')

            if "/beam" in cubes["line_unit"][jj]:
                perbeam = True
            else:
                perbeam = False
            this_spec, this_hdr = sample_at_res(in_data = this_line_file,
                                      ra_samp = samp_ra,
                                      dec_samp = samp_dec,
                                      target_res_as = target_res_as,
                                      target_hdr = ov_hdr,
                                      line_name =cubes["line_name"][jj],
                                      source =this_source,
                                      path_save_fits = data_dir,
                                      save_fits = save_fits,
                                      perbeam = perbeam,
                                      spec_smooth = [spec_smooth,spec_smooth_method])



            this_tag_name = 'SPEC_'+cubes["line_name"][jj].upper()
            this_unit = cubes['line_unit'][jj]
            this_data[this_tag_name] = Column(this_spec , unit= au.Unit(this_unit), description=f'{cubes["line_desc"][jj]} brightness temperature')
            
            #this_line_hdr = fits.getheader(this_line_file)

            sz_this_spec = np.shape(this_spec)
            n_chan = sz_this_spec[1]

            for kk in range(n_pts):
                temp_spec = this_data[this_tag_name][kk]
                temp_spec[0:n_chan] = this_spec[kk,:]
                this_data[this_tag_name][kk] = temp_spec




            #------------------------------------------------------------------
            # Added: Check, if in addition to 3D cube, a customized 2D map is provided

            if not cubes["band_ext"].isnull()[jj]:

                this_band_file = cubes["line_dir"][jj] + this_source + cubes["band_ext"][jj]
                if not quiet:
                    print(f'{"[INFO]":<10}', f'For Cube {cubes["line_name"][jj]} a 2D map is provided.')
                if not path.exists(this_band_file):
                    print(f'{"[ERROR]":<10}', f'Band {cubes["line_name"][jj]} not found for {this_source}.')
                    print(this_band_file)

                    continue


                this_int, this_hdr = sample_at_res(in_data=this_band_file,
                                         ra_samp = samp_ra,
                                         dec_samp = samp_dec,
                                         target_res_as = target_res_as,
                                         target_hdr = ov_hdr,
                                         show = False,
                                         line_name =cubes["line_name"][jj],
                                         source =this_source,
                                         path_save_fits = data_dir,
                                         save_fits = save_fits,
                                         perbeam = perbeam)


                this_tag_name = 'BAND_' + cubes["line_name"][jj].upper()
                this_unit = this_hdr['BUNIT']
                this_data[this_tag_name] = this_int * au.Unit(this_unit)
                

                this_uc_file = cubes["line_dir"][jj] + this_source + str(cubes["band_uc"][jj])
                if not path.exists(this_uc_file):
                    print(f'{"[WARNING]":<10}', f'UC Band {cubes["line_name"][jj]} not found for {this_source}.')
                    continue
                if not quiet:
                    print(f'{"[INFO]":<10}', f'Convolving and sampling line {cubes["line_name"][jj]} for {this_source}.')

                this_uc, this_hdr = sample_at_res(in_data = this_uc_file,
                                        ra_samp = samp_ra,
                                        dec_samp = samp_dec,
                                        target_res_as = target_res_as,
                                        target_hdr = ov_hdr,
                                        perbeam = perbeam,
                                        unc = True)
                this_tag_name = 'EBAND_'+cubes["line_name"][jj].upper()
                this_data[this_tag_name] = this_uc * au.Unit(this_unit)
                
            if not quiet:
                print(f'{"[INFO]":<10}', f'Done with line {cubes["line_name"][jj]}.')


        #---------------------------------------------------------------------
        # SAMPLE MASK
        #---------------------------------------------------------------------

        if len(input_mask) > 0:
            # assign mask file
            this_mask_file = input_mask["mask_dir"][0] + this_source + input_mask["mask_ext"][0]

            # print commands
            if not path.exists(this_mask_file):
                print(f'{"[ERROR]":<10}', f'Mask not found for {this_source}.')
                continue
            print(f'{"[INFO]":<10}', f'Sampling mask for {this_source}.')
        
            # sample mask
            this_spec, this_hdr = sample_mask(in_data = this_mask_file,
                                            ra_samp = samp_ra,
                                            dec_samp = samp_dec,
                                            target_hdr = ov_hdr)

            # add to database
            this_tag_name = 'SPEC_'+input_mask["mask_name"][0].upper()
            this_data[this_tag_name] = Column(this_spec ,unit= au.dimensionless_unscaled, description='User-provided velocity-integration mask (used for integrated products)')
            

            sz_this_spec = np.shape(this_spec)
            n_chan = sz_this_spec[1]

            for kk in range(n_pts):
                temp_spec = this_data[this_tag_name][kk]
                temp_spec[0:n_chan] = this_spec[kk,:]
                this_data[this_tag_name][kk] = temp_spec

            if not quiet:
                print(f'{"[INFO]":<10}', f'Done with mask.')

        
        this_data.write(fname_dict, format='ascii.ecsv', overwrite=True)
    #---------------------------------------------------------------------
    # NOW PROCESS THE SPECTRA
    #---------------------------------------------------------------------
    if not quiet:
        if use_input_mask:
            print(f'{"[INFO]":<10}', 'Start processing spectra; using input mask.')
        else:
            print(f'{"[INFO]":<10}', 'Start processing spectra.')
     
    process_spectra(source_list,
                    cubes,fnames,
                    [NAXIS_shuff, CDELT_SHUFF],
                    run_success,
                    ref_line,
                    SN_processing,
                    strict_mask,
                    input_mask, 
                    use_input_mask, 
                    [mom_thresh,conseq_channels,mom2_method],
                    )
  
    #Open the PyStructure and Save as FITS File
    if save_mom_maps:
        #create a folder to save
        if not os.path.exists(folder_savefits):
            os.makedirs(folder_savefits)
        # Warning
        if spacing_per_beam < 4:
            print(f'{"[WARNING]":<10}', 'Spacing per beam too small for proper resampling to pixel grid.')

        #iterate over the individual sources
        save_mom_to_fits(fnames,
                         cubes,
                         source_list,
                         run_success,
                         overlay_hdr_list,
                         overlay_slice_list,
                         folder_savefits,
                        target_res_as)

    return run_success

#allow input of config file
parser = argparse.ArgumentParser(description="config file")
parser.add_argument("--config")
args, leftovers = parser.parse_known_args()

#check if config file provided
config_prov = False
if not args.config is None:
    print(f'{"[INFO]":<10}', 'Configure file provided.')
    config_prov = True
    conf_file = args.config
    #if folder exists, we delete it first to make sure it contains no files
    if os.path.exists("./Temp_Files/"):
        shutil.rmtree('./Temp_Files')
    os.makedirs("./Temp_Files/")

    temp_f = create_temps(conf_file)
    band_file = temp_f[0]
    cube_file = temp_f[1]
    mask_file = temp_f[2]

    #import and use variables from config_file
    sys.path.append("./Temp_Files/")
    from conf_Py import *


run_success = create_database(conf=config_prov)

#remove the temporary folder after the run is finished
if config_prov:
    shutil.rmtree('./Temp_Files')

if all(run_success):
    print(f'{"[INFO]":<10}', 'Run finished succesfully.')

else:
    print(f'{"[WARNING]":<10}', 'Run terminated with potential critical error!')

#print_warning(0)
