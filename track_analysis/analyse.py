import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import json
from collections import defaultdict as dd
import math
import sys
import os
import time

import imageio

DISPLACEMENTS_FILE_PATH = ("/home/rohan/Dropbox/doherty/experiments/"
                          "11-mapping-sinusoids/data_processed/"
                          "NovDec2019_corrected/displacement_data_200423-1124.csv")

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

# Fuctions for creating plots **************************************************
def mpl_stacked_bar(ax, vals, x_ticks, bar_width):
    """ Creates a stacked bar chart on matplotlib Axis ax"""
    cm = colormap.get_cmap('rainbow', len(vals))
    colors = cm(range(len(vals)))

    if bar_width is None:
        bar_width = x_ticks[1] - x_ticks[0]

    bottoms = np.zeros(len(x_ticks))

    for val, c in zip(sorted(vals, key=np.count_nonzero, reverse=True), colors):

        ax.bar(
            x_ticks, val, bar_width, color=c, bottom=bottoms
        )
        bottoms += val

def ordinal_histogram(df, attr_name, xlabel="", ylabel="", title="", xtick_labels=None, path=None):
    
    x_ticks = df[attr_name].dropna().unique()
    x_ticks.sort()
    
    exp_counts = []
    #pylint:disable=unused-variable
    for group_name, group in df.groupby([EXPNAME, DATE]):

        counts = np.empty(len(x_ticks))
        for i, t in enumerate(x_ticks):
            
            counts[i] = group[group[attr_name] == t].shape[0]
        exp_counts.append(counts)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    

    bar_width = x_ticks[1] - x_ticks[0]

    mpl_stacked_bar(ax, exp_counts, x_ticks, bar_width)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=x_ticks.min() - bar_width / 2)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)

def numeric_histogram(df, attr_name, xlabel="", ylabel="", title="", nbins=30, path=None, xlims=None, ylims=None):

    x_min = df[attr_name].min()
    x_max = df[attr_name].max()
    
    bucket_size = (x_max - x_min) / nbins

    #convert a df value to the its bucket index
    to_bucket_index = lambda z : math.floor((z - x_min) / bucket_size)

    #convert a bucket index to the center of that bucket
    bucket_index_to_center = lambda i : i * bucket_size + bucket_size / 2 + x_min

    #x_ticks are the center of each bucket
    x_ticks = np.array([bucket_index_to_center(i) for i in range(nbins)]) 

    #the buckets for each experiment
    all_buckets = []

    #pylint:disable=unused-variable
    for group_name, group in df.groupby(EXP_ID):
        buckets = np.zeros(nbins)

        for x in group[attr_name].dropna():
            if x == x_max: x -= 1e-13
            buckets[to_bucket_index(x)] += 1

        all_buckets.append(buckets)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    mpl_stacked_bar(ax, all_buckets, x_ticks, bucket_size)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=x_ticks.min() - (bucket_size / 2))
    
    if xlims is not None:
        ax.set_xlim(left=xlims[0], right=xlims[1])

    if ylims is not None:
        ax.set_ylim(left=ylims[0], right=ylims[1])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)

def scatter_means(df, x_attr, y_attr, xlabel="", ylabel="", title="", path=None, x_max=None):

    if x_max is not None:
        df = df[df[x_attr] <= x_max]

    x_axis = df[x_attr].unique()
    x_axis.sort()

    y_axis = np.empty(len(x_axis))
    for i, x in enumerate(x_axis):

        y_axis[i] = df[df[x_attr] == x][y_attr].mean()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(x_axis, y_axis)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)

def turning_ratio_scatter(df, xlabel="", ylabel="", title="", tmax=None, path=None):

    if tmax is not None:
        df = df[df[TIMESTEP] <= tmax]

    timesteps = df[TIMESTEP].unique()


    turning_ratio = np.empty(len(timesteps))

    for i, t in enumerate(timesteps):
        df_at_t = df[df[TIMESTEP] == t]
        turning_ratio[i] = np.count_nonzero(df_at_t[DIRECTION] == -1) / df_at_t[DIRECTION].count()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(timesteps, turning_ratio)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)


def image_file_sequence_to_gif(imseq, gifpath, frame_duration=0.5):
    import imageio
    with imageio.get_writer(gifpath, mode='I', duration=frame_duration) as writer:
        for filename in imseq:
            image = imageio.imread(filename)
            writer.append_data(image)


def to_output_dir():

    name = time.strftime("%y%m%d-%H%M") + "_" + "displacement-figures"

    os.chdir(PARENT_OUTDIR)
    os.makedirs(name)
    os.chdir(name)

def generate_all_plots():

    max_time = 250

    base_timestep = int(sys.argv[1])

    if base_timestep == 10:
        times = [10, 20]
    else:
        times = [30, 60, 90, 120]

    disp_df = pd.read_csv(DISPLACEMENTS_FILE_PATH, dtype=DTYPE_DICT)
    
    disp_df = filter_displacement_data(disp_df, base_timestep=base_timestep)
    
    disp_sq = disp_df[DISP] ** 2
    disp_sq.name = DISP_SQ
    disp_df = disp_df.join(disp_sq)

    disp_strt_sq = disp_df[STRT_DISP] ** 2
    disp_strt_sq.name = STRT_DISP_SQ
    disp_df = disp_df.join(disp_strt_sq)

    signed_disp = disp_df[DISP] * disp_df[DIRECTION]
    signed_disp.name = SIGNED_DISP
    disp_df = disp_df.join(signed_disp)

    disp_array = disp_df[DISP].to_numpy()
    total = len(disp_array)
    zero_count = total - np.count_nonzero(disp_array)
    pc = round(100 * zero_count / total, ndigits=2)

    print(f"{zero_count} of {total} are zero ({pc}%).")

    
    to_output_dir()

    #timestep histogram
    ordinal_histogram(
        disp_df, 
        TIMESTEP, 
        title="Number of observations per timestep",
        xlabel="Change in time (seconds)", 
        ylabel="Number of observations",
        path=f"obs-per-timestep-bt{base_timestep}.svg"
    )

    for time in times:
        
        t_df = disp_df[disp_df[TIMESTEP] == time]

        #turning histogram
        ordinal_histogram(
            t_df, 
            DIRECTION, 
            title=f"Turning frequency for time change {time} sec", 
            xtick_labels=['Reverse', 'Forward'],
            ylabel="Number of observations",
            path=f"turning-hist-t{time}.svg"
        )

        #raw distance histogram
        numeric_histogram(
            t_df, 
            DISP, 
            title=f"Absolute displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=f"abs-disp-hist-t{time}.svg"
        )

        #straight-line histogram
        numeric_histogram(
            t_df, 
            STRT_DISP, 
            title=f"Displacement (straight line) histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=f"abs-disp-strt-hist-t{time}.svg"
        )

        #signed distance histogram
        numeric_histogram(
            t_df, 
            SIGNED_DISP, 
            title=f"Signed displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=f"sgn-disp-hist-t{time}.svg",
        )

        """
        numeric_histogram(
            t_df, 
            DISP, 
            title=f"Absolute displacement histogram for time change {time} sec (zoomed in)", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            xlims=[0, 4],
            nbins=300,
            path=f"abs-disp-hist-t{time}-zoomed.svg"
        )
         

        numeric_histogram(
            t_df, 
            SIGNED_DISP, 
            title=f"Signed displacement histogram for time change {time} sec (zoomed in)", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            xlims=[-4, 4], 
            nbins=300, 
            path=f"sgn-disp-hist-t{time}-zoomed.svg"
        )
        """

    #turning ratio scatter
    turning_ratio_scatter(
        disp_df, 
        tmax=max_time,
        title="Probability of direction change vs. Time", 
        ylabel="Probability of reversing", 
        xlabel="Change in time (seconds)",
        path=f"turning-ratio-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        TIMESTEP, 
        DISP,
        title="Mean displacement vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"mean-disp-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        TIMESTEP, 
        DISP_SQ,
        title="Mean-squared displacement vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"mean-sq-disp-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        TIMESTEP, 
        STRT_DISP,
        title="displacement (straight line) vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"disp-strt-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        TIMESTEP, 
        STRT_DISP_SQ,
        title="Mean-squared displacement (striaght line) vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"mean-sq-disp-strt-scatter-bt{base_timestep}.svg"    
    )

    #making gifs

    os.mkdir("temp")
    os.chdir("temp")
    time = base_timestep
    t_df = disp_df[disp_df[TIMESTEP] == time]

    ad_hist_files = []
    sd_hist_files = []
    strt_hist_files = []

    while time <= 800:
        
        #raw distance histogram
        ad_name = f"abs-disp-hist-t{time}.jpg"
        ad_hist_files.append(ad_name)
        numeric_histogram(
            t_df, 
            DISP, 
            title=f"Absolute displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=ad_name, 
            xlims=[0, 100]
        )

        #raw distance histogram
        sd_name = f"sgn-disp-hist-t{time}.jpg"
        sd_hist_files.append(sd_name)
        numeric_histogram(
            t_df, 
            SIGNED_DISP, 
            title=f"Signed displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=sd_name, 
            xlims=[-100, 100]
        )

        #raw distance histogram
        srt_name = f"strt-disp-hist-t{time}.jpg"
        strt_hist_files.append(srt_name)
        numeric_histogram(
            t_df, 
            STRT_DISP, 
            title=f"Displacement histogram (straight line) for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=srt_name, 
            xlims=[0, 100]
        )

        t_df = disp_df[disp_df[TIMESTEP] == time]
        time += base_timestep

    image_file_sequence_to_gif(ad_hist_files, f"../ad-hist-b{base_timestep}.gif")
    image_file_sequence_to_gif(sd_hist_files, f"../sd-hist-b{base_timestep}.gif")
    image_file_sequence_to_gif(strt_hist_files, f"../strt-hist-b{base_timestep}.gif")

    return disp_df

def means_df(df, x_attr, y_attr):

    x_axis = df[x_attr].unique()
    x_axis.sort()

    y_axis = np.empty(len(x_axis))
    for i, x in enumerate(x_axis):

        y_axis[i] = df[df[x_attr] == x][y_attr].mean()

    return pd.DataFrame(np.array([x_axis, y_axis]).T)
    
def time_statistics_df(df):

    x_axis = df[TIMESTEP].unique()
    x_axis.sort()

    y_axis_mean_disp = np.empty(len(x_axis))
    y_axis_mean_sq_disp = np.empty(len(x_axis))
    y_axis_mean_strt_disp = np.empty(len(x_axis))
    y_axis_mean_sq_strt_disp = np.empty(len(x_axis))
    for i, x in enumerate(x_axis):

        y_axis_mean_disp[i] = df[df[TIMESTEP] == x][DISP].mean()
        y_axis_mean_sq_disp[i] = df[df[TIMESTEP] == x][DISP_SQ].mean()
        y_axis_mean_strt_disp[i] = df[df[TIMESTEP] == x][STRT_DISP].mean()
        y_axis_mean_sq_strt_disp[i] = df[df[TIMESTEP] == x][STRT_DISP_SQ].mean()

    return pd.DataFrame(
        np.array(
            [x_axis, 
            y_axis_mean_disp, 
            y_axis_mean_sq_disp, 
            y_axis_mean_strt_disp, 
            y_axis_mean_sq_strt_disp]).T,
        columns=["timestep (s)", "mean displacememt (mu m)", "mean-squared displacement (mu m ^2)", "mean straight displacement (mu m)", "mean-squared straight displacement (mu m ^2)"]
        )

if __name__ == "__main__":


    OUTPUT = "/home/rohan/Desktop/stats.csv"

       
    
    
    generate_all_plots()
    
    """
    df = pd.read_csv(DISPLACEMENTS_FILE_PATH, dtype=DTYPE_DICT)
    df = filter_displacement_data(df, base_timestep=None)

    disp_sq = df[DISP] ** 2
    disp_sq.name = DISP_SQ
    df = df.join(disp_sq)

    disp_strt_sq = df[STRT_DISP] ** 2
    disp_strt_sq.name = STRT_DISP_SQ
    df = df.join(disp_strt_sq)
    
    stats = time_statistics_df(df)

    stats.to_csv(OUTPUT)

    """

    

