import numpy as np
import json

DISPLACEMENTS_FILE_PATH = ("/home/rohan/Dropbox/doherty/experiments/"
                          "11-mapping-sinusoids/data_processed/"
                          "NovDec2019_corrected/displacement_data_no_endpoints_200714-1617.csv")

BASE_TIMESTEP_PATH = ("/home/rohan/Dropbox/doherty/experiments/"
                     "11-mapping-sinusoids/data_processed/"
                     "NovDec2019_corrected/base_timesteps.json")

PARENT_OUTDIR = "/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/data_processed"

#the names of attributes used when outputing data
DISP = 'curve_displacement (micro-m)'
DISP_ERROR = 'abs_displacement_numeric_error'
TIMESTEP = 'time_step'
TID = 'track_name'
EXPNAME = 'experiment'
DATE = 'date'
TRACK_NUM_POINTS = 'num_points'
TRACK_DURATION = 'duration (sec)'
DIRECTION = "reldir"
STRT_DISP = 'straight displacement (micro-m)'

EXP_ID = "experiment_id"
DISP_SQ = "disp_sq"
SIGNED_DISP = "signed_disp"
STRT_DISP_SQ = "straight disp sq"

DTYPE_DICT = {
    DISP: np.float64, DISP_ERROR: np.float64, TIMESTEP: np.float64, 
    TID: str, EXPNAME: str, DATE: str, TRACK_NUM_POINTS: np.int64, 
    TRACK_NUM_POINTS: np.int64, TRACK_DURATION: np.float64, DIRECTION: np.float64, 
    STRT_DISP: np.float64}

#criteria to exclude and include data

#minimum total track displacement (um)
DISP_THRESHOLD = 30

#minimum number of track points
NUM_POINTS_THRESHOLD = 5

#minimum observation period of track (seconds)
DURATION_THRESHOLD = 60

#equivalent for simulated cells
SIM_DURATION_THRESHOLD = 6


#dates of experiments to analyse
DATES = {"191205", "191206"}

# Functions for loading and filtering data *************************************
def get_base_timestep_dicts(path=BASE_TIMESTEP_PATH):
    """ Returns a dictionary keyed by experiment id that holds the base timestep
        for that experiment"""
    with open(path) as f:
        return {float(t): eid for t, eid in json.load(f).items()}

def filter_short_tracks(df, disp_threshold):
    """ Removes data from tracks with less total displacement that disp_threshold"""
    idxs = [
        idx for idx in df.groupby([DATE, EXPNAME, TID]).groups.values() 
        if df.loc[idx][DISP].max() >= disp_threshold
    ]
    
    i = idxs[0]
    i = i.append(idxs[1:])
    return df.loc[i]

def filter_displacement_data(disp_df, base_timestep=None, remove_zeros=False):
    """ Function which defines which data is filtered prior to analysis"""

    #remove data collected on dates we are not interested in
    disp_df = disp_df[disp_df[DATE].isin(DATES)]

    if remove_zeros:
        disp_df = disp_df[disp_df[DISP] != 0]

    #remove data from tracks shorter than DISP_THRESHOLD
    disp_df = filter_short_tracks(disp_df, DISP_THRESHOLD)

    #remove data from tracks with less than NUM_POINTS_THRESHOLD observations
    disp_df = disp_df[disp_df[TRACK_NUM_POINTS] >= NUM_POINTS_THRESHOLD]

    #remove data from tracks which were observed for less time than DURATION_THRESHOLD
    disp_df = disp_df[disp_df[TRACK_DURATION] >= DURATION_THRESHOLD]

    #calculate an experiment id from date and experiment name collumns
    exp_id = disp_df[DATE] + "/" + disp_df[EXPNAME]
    exp_id.name = EXP_ID

    disp_df = disp_df.join(exp_id)
    if base_timestep is None:
        return disp_df
    

    disp_df = disp_df[disp_df[TIMESTEP] % base_timestep == 0]

    expid_base_ts = get_base_timestep_dicts()

    allowed_expids = set()
    for bts, exp_ids in expid_base_ts.items():
        if base_timestep % bts == 0:
            allowed_expids = allowed_expids.union(exp_ids)

    return disp_df[disp_df[EXP_ID].isin(allowed_expids)]

def filter_simulated_displacement_data(disp_df, remove_zeros=False):
    """ Function which defines which data is filtered prior to analysis"""

    if remove_zeros:
        disp_df = disp_df[disp_df[DISP] != 0]

    #remove data from tracks shorter than DISP_THRESHOLD
    #disp_df = filter_short_tracks(disp_df, disp_threshold)

    #remove data from tracks with less than NUM_POINTS_THRESHOLD observations
    disp_df = disp_df[disp_df[TRACK_NUM_POINTS] >= NUM_POINTS_THRESHOLD]

    #remove data from tracks which were observed for less time than DURATION_THRESHOLD
    disp_df = disp_df[disp_df[TRACK_DURATION] >= SIM_DURATION_THRESHOLD]

    #calculate an experiment id from date and experiment name collumns
    exp_id = disp_df[DATE] + "/" + disp_df[EXPNAME]
    exp_id.name = EXP_ID

    disp_df = disp_df.join(exp_id)
    
    return disp_df
 