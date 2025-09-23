"""
This routine generates a dictionary, similar to  the struct in idl.
Several side functions are part of this routine.
The original routine was wrirtten in idl (see create_database.pro)

The output is a dictionary saved as an .npy file. To open it in a new python
script use for example:

    > read_dictionary = np.load('datafile.npy',allow_pickle = True).item()
    > ra_samp = read_dictionary["ra_deg"]
    > dec_samp = read_dictionary["dec_deg"]
    > intensity = np.nansum(read_dictionary["SPEC_VAL_CO21"], axis = 0)
    > plt.scatter(ra_samp, dec_samp, c = intensity, marker = "h")

MODIFICATION HISTORY
    -   v1.0.1 16-22 October 2019: Conversion from IDL to Python
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


"""
__author__ = "J. den Brok"
__version__ = "v3.1.1"
__email__ = "jakob.den_brok@cfa.harvard.edu"
__credits__ = ["L. Neumann","M. Jimenez-Donaire", "E. Rosolowsky","A. Leroy ", "I. Beslic"]


import numpy as np
import pandas as pd
import os.path
from os import path
import shutil
from astropy.io import fits
from datetime import date, datetime
import re
import argparse
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
from making_axes import make_axes  # LN: not actually used 
from processing_spec import process_spectra
from message_list import *
from save_moment_maps import save_mom_to_fits

#----------------------------------------------------------------------
# Change these lines of code with correct directory and names
#----------------------------------------------------------------------

# <path to directory with the data files>
data_dir = "data/"

# <filename of geometry file>
geom_file = "List_Files/geometry.txt"
# <filename of band file>
band_file = "List_Files/band_list.txt"
# <filename of cube file>
cube_file = "List_Files/cube_list.txt"
# <filename of overlay or mask> #should be stored in data_dir
overlay_file = "_12co21.fits"

# <Output Directory for Dictionaries>
out_dic = "Output/"

# Set the target resolution for all data in arcseconds (if resolution set to angular)
target_res = 27.


#!!!!!!!!!!!!!Advanced------------------------------------------
NAXIS_shuff = 200
CDELT_SHUFF = 4000.  #m/s
spacing_per_beam = 2 #default, use half beam spacing
# give number (in units deg) or set to "auto"
max_rad = "auto" #default extension of the map in deg (increase, if you map is larger)

"""
angular: use target_res in as
physical: convert target_res (in pc) to as
native: use the angular resolution of the overlay image
"""
resolution = 'angular'

# Save the convolved cubes & bands
save_fits = False

"""
Define which line to use as reference line for the spectral processing
"first": use first line in cube_list as reference line
"<LINE_NAME>": Use line name as reference line
"all": Use all lines in cube for mask
n: (integer) use first n lines as reference. n=0 is same result as "first".
"ref+HI": Use first line and HI
"""
ref_line = "first"

#define upper and lower mask threshold (S/N)
SN_processing = [2,4]
strict_mask= False
#define SN threshold for Mom1, Mom2 and EW calculation (for individual lines)
mom_thresh = 3
conseq_channels = 3 #needs to integer more than 3
#differentiate between "fwhm", "sqrt", or "math"
# math: use mathematical definition
# sqrt: take square-root of mom2
# fwhm: convert sqrt(mom2) to fwhm
mom2_method = "fwhm"


"""
Spectral smoothing

"default": Do not perform any spectral smoothing
"overlay": Perform spectral smoothing to spectral resolution of overlay cube
n: float – convolve to spectral resolution n [km/s]
"""
spec_smooth = "default"

"""
define the way the spectral smoothing should be performed:
"binned": binn channels together (to nearest integer of ratio theta_target/theta_nat)
"gauss": perform convolution with gaussian kernel (theta_target^2-theta_nat^2)**0.5
!!!! Warning, gaussian smoothing seems to systematicaly underestimate the rms by 10-15%
"combined": do the binned smoothing first (to nearest integer ratio) and then the rest via Gauss
"""
spec_smooth_method = "binned"

"""
Define how pyStructure should be run
- overwrite: A file will be created and overwritten if the PyStructure code is run
- fill: When running the PyStructure code, opens an exisitng PyStructure and completes it with additional lines
- archive: each run creates a new copy, ensuring archiving of all runs (not yet implemented)
"""
structure_creation ="overwrite"


"""
Save the created moment maps as fits file
"""
save_mom_maps = False

#folder to save fits files in
folder_savefits="./saved_FITS_files/"
#---------------------------------------------------------------


#----------------------------------------------------------------------
# The function that generates an empty directory
#----------------------------------------------------------------------
def empire_record_header():
    """
    Make the first, general fields for an EMPIRE database record.
    """

    new_structure_empire = {
    "source": '',
    "ra_deg": None,
    "dec_deg": None,
    "dist_mpc": None,
    "posang_deg": None,
    "incl_deg": None,
    "beam_as": None,
    "rgal_as": None,
    "rgal_kpc": None,
    "rgal_r25": None,
    "theta_rad": None
    }

    return new_structure_empire

def fill_checker(fname, sample_coord, bands, cubes):
    """
    Function that checks if a given PyStructure exists and matches in terms of the sampling points.
    :param fname: PyStructure Filename
    :param sample_coord: [samp_ra, samp_dec] - The sample coordinates. Used to match to existing PySturcture.
    :param bands:
    :param cubes
    """

    this_data = np.load(fname,allow_pickle = True).item()

    #Check 1: Enusre that the coordinates are identical (1e-12 to allow wiggle room due to rounding errors)
    if abs(np.nansum(this_data['ra_deg']-sample_coord[0])) + abs(np.nansum(this_data['dec_deg']-sample_coord[1]))>1e-12:
        raise ValueError('The PyStructure does not match. Please run code setting the "structure_creation" key to "overwrite"')
    
    #Check 2: Now check which bands and cubes 
    fill_bands = []
    for band_nm in bands["band_name"]:
        if f'INT_VAL_{band_nm.upper()}' in  list(this_data.keys()):
            fill_bands.append(band_nm)
    fill_cubes = []
    for cube_nm in cubes["line_name"]:
        if f'INT_VAL_{cube_nm.upper()}' in  list(this_data.keys()):
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
    :param just_source: String name of a source, if one wants only one galaxy
    :param quiet: Verbosity set to mute
    :param conf: Config File provided
    :return database: python dictionary
    """

    if quiet == False:
        print(f'{"[INFO]":<10}', 'Reading in galaxy parameters.')
    names_glxy = ["galaxy", "ra_ctr", "dec_ctr", "dist_mpc", "e_dist_mpc",
                  "incl_deg", "e_incl_deg","posang_deg", "e_posang_deg",
                  "r25", "e_r25"]
    glxy_data = pd.read_csv(geom_file, sep = "\t",names = names_glxy,
                            comment = "#")

    #define list of sources (need to differentiate between conf file input and default)
    if conf:
        if isinstance(sources, tuple):
            galaxy_list = list(sources)
        else:
            galaxy_list = [sources]

    else:
        galaxy_list = list(glxy_data["galaxy"])

    n_sources = len(galaxy_list)
    # -----------------------------------------------------------------
    # GENERATE THE EMPTY DATA STRUCTURE
    # -----------------------------------------------------------------
    if quiet == False:
        print(f'{"[INFO]":<10}', 'Generating (new) dictionary.')
    empty_structure = empire_record_header()

    # Add the bands to the structure
    band_columns = ["band_name","band_desc", "band_unit",
                    "band_ext", "band_dir","band_uc" ]
    bands = pd.read_csv(band_file, names = band_columns, sep='[\s,]{2,20}', comment="#")

    n_bands = len(bands["band_name"])
    for ii in range(n_bands):
        empty_structure = add_band_to_struct(struct=empty_structure,
                                         band=bands["band_name"][ii],
                                         unit=bands["band_unit"][ii],
                                         desc=bands["band_desc"][ii])

    if quiet == False:
        print(f'{"[INFO]":<10}', f'{n_bands} band(s) loaded into structure.')


    # Add the cubes to the structure
    cube_columns = ["line_name", "line_desc", "line_unit", "line_ext", "line_dir" , "band_ext", "band_uc"]

    cubes = pd.read_csv(cube_file, names = cube_columns, sep='[\s,]{2,20}', comment="#")
    n_cubes = len(cubes["line_name"])
    for ii in range(n_cubes):
        empty_structure = add_spec_to_struct(struct=empty_structure,
                                         line=cubes["line_name"][ii],
                                         unit=cubes["line_unit"][ii],
                                         desc=cubes["line_desc"][ii])

        # if we provide a cube for which we already have the 2D map, include it as a band
        if not cubes["band_ext"].isnull()[ii]:
            empty_structure = add_band_to_struct(struct=empty_structure,
                                                    band=cubes["line_name"][ii],
                                                    unit=cubes["line_unit"][ii]+"km/s",
                                                    desc=cubes["line_desc"][ii])

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
        empty_structure = add_spec_to_struct(struct=empty_structure,
                                            line=input_mask["mask_name"][0],
                                            desc=input_mask["mask_desc"][0])
        # remove unit for mask
        del empty_structure[f'SPEC_UNIT_{input_mask["mask_name"][0].upper()}']

        if quiet == False:
            if use_input_mask:
                print(f'{"[INFO]":<10}', f'Input mask loaded into structure; will be used for products.')
            else:
                print(f'{"[INFO]":<10}', f'Input mask loaded into structure; will NOT be used for products.')


    #-----------------------------------------------------------------
    # LOOP OVER SOURCES
    #-----------------------------------------------------------------

    #additional parameters
    run_success = [True]*n_sources #keep track if run succesfull for each galaxy
    fnames=[""]*n_sources   #filename save for galaxy
    overlay_hdr_list = []
    overlay_slice_list = []

    for ii in range(n_sources):
        #if config file provided, use the list of galaxies provided therein

        this_source = galaxy_list[ii]

        if not this_source in list(glxy_data["galaxy"]):
            run_success[ii]=False

            print(f'{"[ERROR]":<10}', f'{this_source} not in galaxy table.')

            continue

        #assign correct index of list and input galaxy (relevant for index file)
        ii_list = np.where(np.array(glxy_data["galaxy"])==this_source)[0][0]


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

        this_vaxis_ov = make_axes(ov_hdr, vonly = True) # LN: variable not used
        #mask = total(finite(hcn_cube),3) ge 1
        mask = np.sum(np.isfinite(ov_cube), axis = 0)>=1
        mask_hdr = twod_head(ov_hdr)
        overlay_hdr_list.append(mask_hdr)
        if resolution == 'native':
            target_res_as = np.max([ov_hdr['BMIN'], ov_hdr['BMAJ']]) * 3600
        elif resolution == 'physical':
            target_res_as = 3600 * 180/np.pi * 1e-6 * target_res / glxy_data['dist_mpc'][ii_list]
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
        fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+'.npy'

        if "archive" in structure_creation:
            #check if basic file already exists. Otherwise, start with version numbering
            if os.path.exists(fname_dict):
                file_version=1
                fname_dict = fname_dict[:-4]+f"_v{file_version}.npy"
                while os.path.exists(fname_dict):
                    file_version+=1
                    fname_dict = out_dic+this_source+"_data_struct_"+res_suffix+'_'+date_str+f'_v{file_version}.npy'
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
                             ra_ctr = glxy_data["ra_ctr"][ii_list],
                             dec_ctr = glxy_data["dec_ctr"][ii_list],
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
            this_data = {}

            #for n in range(n_pts):
            for key in empty_structure.keys():
                    #this_data.setdefault(key, []).append(empty_structure[key])
                    this_data[key]=empty_structure[key]

            this_tag_name = 'SPEC_VCHAN0'
            this_data[this_tag_name] = ov_hdr["CRVAL3"]
            this_tag_name = 'SPEC_DELTAV'
            this_data[this_tag_name] = ov_hdr["CDELT3"]
            this_tag_name = 'SPEC_CRPIX'
            this_data[this_tag_name] = ov_hdr["CRPIX3"]
            # Some basic parameters for each galaxy:
            this_data["source"] = this_source
            this_data["ra_deg"] = samp_ra
            this_data["dec_deg"] = samp_dec
            this_data["dist_mpc"] = glxy_data["dist_mpc"][ii_list]
            this_data["posang_deg"] = glxy_data["posang_deg"][ii_list]
            this_data["incl_deg"] = glxy_data["incl_deg"][ii_list]
            this_data["beam_as"] = target_res_as

            # Convert to galactocentric cylindrical coordinates
            rgal_deg, theta_rad = deproject(samp_ra, samp_dec,
                                        [glxy_data["posang_deg"][ii_list],
                                         glxy_data["incl_deg"][ii_list],
                                         glxy_data["ra_ctr"][ii_list],
                                         glxy_data["dec_ctr"][ii_list]
                                        ], vector = True)


            this_data["rgal_as"] = rgal_deg * 3600
            this_data["rgal_kpc"] = np.deg2rad(rgal_deg)*this_data["dist_mpc"]*1e3
            this_data["rgal_r25"] = rgal_deg/(glxy_data["r25"][ii_list]/60.)
            this_data["theta_rad"] = theta_rad

        #---------------------------------------------------------------------
        # LOOP OVER MAPS, CONVOLVING AND SAMPLING
        #---------------------------------------------------------------------

        for jj in range(n_bands):
            if 'fill' in structure_creation:
                if bands["band_name"][jj] in fill_bands:
                    continue
                #need to add an entry into the PyStructure
                else:
                    this_data = add_band_to_struct(struct=this_data,
                                         band=bands["band_name"][jj],
                                         unit=bands["band_unit"][jj],
                                         desc=bands["band_desc"][jj])
                    

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
                                     galaxy =this_source,
                                     path_save_fits = data_dir,
                                     save_fits = save_fits,
                                     perbeam = perbeam)


            this_tag_name = 'INT_VAL_' + bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_int
            else:
                print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                continue

            #; MJ: I AM ADDING THE CORRESPONDING UNITS

            this_unit = bands["band_unit"][jj]
            this_tag_name = 'INT_UNIT_' + bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_unit
            else:
                print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                continue

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
            this_tag_name = 'INT_UC_'+bands["band_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_uc
            else:
                print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                continue


        #---------------------------------------------------------------------
        # LOOP OVER CUBES, CONVOLVING AND SAMPLING
        #---------------------------------------------------------------------

        for jj in range(n_cubes):

            if 'fill' in structure_creation:
                if cubes["line_name"][jj] in fill_cubes:
                    continue
                else:
                    this_data = add_spec_to_struct(struct=this_data,
                                         line=cubes["line_name"][jj],
                                         unit=cubes["line_unit"][jj],
                                         desc=cubes["line_desc"][jj])

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
                                      galaxy =this_source,
                                      path_save_fits = data_dir,
                                      save_fits = save_fits,
                                      perbeam = perbeam,
                                      spec_smooth = [spec_smooth,spec_smooth_method])



            this_tag_name = 'SPEC_VAL_'+cubes["line_name"][jj].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_spec
            else:
                print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                continue

            #this_line_hdr = fits.getheader(this_line_file)

            this_vaxis = make_axes(this_hdr, vonly = True) # LN: variable not used
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
                                         galaxy =this_source,
                                         path_save_fits = data_dir,
                                         save_fits = save_fits,
                                         perbeam = perbeam)


                this_tag_name = 'INT_VAL_' + cubes["line_name"][jj].upper()
                if this_tag_name in this_data:
                    this_data[this_tag_name] = this_int
                else:
                    print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                    continue


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
                this_tag_name = 'INT_UC_'+cubes["line_name"][jj].upper()
                if this_tag_name in this_data:
                    this_data[this_tag_name] = this_uc
                else:
                    print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name}to the database.')
                    continue
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
            this_tag_name = 'SPEC_VAL_'+input_mask["mask_name"][0].upper()
            if this_tag_name in this_data:
                this_data[this_tag_name] = this_spec
            else:
                print(f'{"[ERROR]":<10}', f'I had trouble matching tag {this_tag_name} to the database.')
                continue

            # this_vaxis = make_axes(this_hdr, vonly = True) # LN: variable not used
            sz_this_spec = np.shape(this_spec)
            n_chan = sz_this_spec[1]

            for kk in range(n_pts):
                temp_spec = this_data[this_tag_name][kk]
                temp_spec[0:n_chan] = this_spec[kk,:]
                this_data[this_tag_name][kk] = temp_spec

            if not quiet:
                print(f'{"[INFO]":<10}', f'Done with mask.')

        
        np.save(fname_dict, this_data)

    #---------------------------------------------------------------------
    # NOW PROCESS THE SPECTRA
    #---------------------------------------------------------------------
    if not quiet:
        if use_input_mask:
            print(f'{"[INFO]":<10}', 'Start processing spectra; using input mask.')
        else:
            print(f'{"[INFO]":<10}', 'Start processing spectra.')
            
    process_spectra(glxy_data,
                    galaxy_list,
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
                         galaxy_list,
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
