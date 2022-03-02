""" 
This module stores configuration data, such as file paths to data files.

"""

import os
import sys


# File paths and file names ****************************************************

#this is a configuration file which determines from where data is read and saved
working_dir = "/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/"

#the subdirectory of working_dir where all input data is stored
input_dir = "./data/NovDec2019_corrected/"

#name of the file storing time data
time_fname = "time.time"

#name of the file storing spots data
spots_fname = "Spots 1.npz"

#file extension of surface files
surf_ext = ".npz"

#relative file paths of surfaces to exclude from processing (relative to inpud_dir)
surfs_to_exclude = {
    "./191127/Mouse 2 CD31 alexa647 3 _drift_corrected/1000001053.npz", 
    "./191127/Mouse 2 CD31 alexa647 3 _drift_corrected/1000001015.npz", 
    "./191127/Mouse 2 CD31 alexa647 3 _drift_corrected/1000062087_1000024114.npz", 
    "./191127/Mouse 2 CD31 alexa647 3 _drift_corrected/1000019960.npz",
    "./191127/Mouse 2 CD31 alexa647 3 _drift_corrected/1000012547.npz",
    "./191129/mouse5 qdot 3/Surfaces 1.npz",
    "./191129/Mouse 5 CD31 647 2 _corrected_drift/Surfaces 1.npz"
    }

#seperator between track ids in surface filenames
tid_separator = "_"

#file to store parameterisation data
param_file = "param_axes.json"

#for testing program with just one surface / spots
test_surf = "/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/data/NovDec2019_corrected/191205/mouse1_4/1000055324.npz"

test_spots = "/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/data/NovDec2019_corrected/191206/mouse1_4_30sec/Spots 1.npz"

# Function to iterate over the data *******************************************

def data_iter(spots_fname=spots_fname, time_fname=time_fname, surf_ext=surf_ext, surfs_to_exclude=surfs_to_exclude):
    for pwd, dirs, files in os.walk("."):
        
        if dirs:
            continue
            
        if time_fname not in files:
            print(f"Warning: No time file in {pwd}. Skipping this directory.", file=sys.stderr)
            continue
        
        with open(os.path.join(pwd, time_fname)) as t:
            time_step = float(next(t))

        if spots_fname not in files:
            print(f"Warning: No spots file in {pwd}. Skipping this directory.", file=sys.stderr)
            continue
        
        spots_file = os.path.join(pwd, spots_fname)

        fullname = lambda f : os.path.join(pwd, f)

        #collect only surface files
        surfs = []
        for f in files:

            if f == spots_fname:
                continue

            if f == time_fname:
                continue

            pth, ext = os.path.splitext(f)
            if ext != surf_ext:
                continue
            
            f = fullname(f)
            if f in surfs_to_exclude:
                continue
            
            surfs.append(f)

        if surfs:
            yield time_step, spots_file, surfs

def track_ids(surface_filepath, tid_separator=tid_separator):
    """ Gets the track ids corresponding to the given surface."""
    #extract file name from path
    #pylint: disable=unused-variable
    d, surface_filename = os.path.split(surface_filepath)

    #remove file extension
    #pylint: disable=unused-variable
    surface_name, s, e =  surface_filename.rpartition(".") 
  
    return [int(tid) 
            for tid in surface_name.split(tid_separator) 
            if str.isdigit(tid)]

def generate_track_name(tid, spots_file):
    """ Use this method to generate names for tracks based on the track id and 
        name of the file it comes from."""
    pth, fname = os.path.split(spots_file) #pylint: disable=unused-variable
    return pth + "_" + tid

def get_experiment(pth):
    pth, fname = os.path.split(pth) #pylint: disable=unused-variable
    pth, exp_name = os.path.split(pth)
    pth, datestr = os.path.split(pth)

    return datestr, exp_name
