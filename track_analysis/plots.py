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
import argparse

import io_tools

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
    for group_name, group in df.groupby([io_tools.EXPNAME, io_tools.DATE]):

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

    if bucket_size == 0:
        print(x_max, x_min)
        print(path)
        return

    #convert a df value to the its bucket index
    to_bucket_index = lambda z : math.floor((z - x_min) / bucket_size)

    #convert a bucket index to the center of that bucket
    bucket_index_to_center = lambda i : i * bucket_size + bucket_size / 2 + x_min

    #x_ticks are the center of each bucket
    x_ticks = np.array([bucket_index_to_center(i) for i in range(nbins)]) 

    #the buckets for each experiment
    all_buckets = []

    #pylint:disable=unused-variable
    for group_name, group in df.groupby(io_tools.EXP_ID):
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
        df = df[df[io_tools.TIMESTEP] <= tmax]

    timesteps = df[io_tools.TIMESTEP].unique()


    turning_ratio = np.empty(len(timesteps))

    for i, t in enumerate(timesteps):
        df_at_t = df[df[io_tools.TIMESTEP] == t]
        turning_ratio[i] = np.count_nonzero(df_at_t[io_tools.DIRECTION] == -1) / df_at_t[io_tools.DIRECTION].count()

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

    os.chdir(io_tools.PARENT_OUTDIR)
    os.makedirs(name)
    os.chdir(name)

def generate_all_plots(disp_df, base_timestep):


    if base_timestep == 1:
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12   ]
        max_time = 12

    elif base_timestep == 10:
        times = [10, 20]
        max_time = 250
    else:
        times = [30, 60, 90, 120]
        max_time = 250

    
    disp_sq = disp_df[io_tools.DISP] ** 2
    disp_sq.name = io_tools.DISP_SQ
    disp_df = disp_df.join(disp_sq)

    disp_strt_sq = disp_df[io_tools.STRT_DISP] ** 2
    disp_strt_sq.name = io_tools.STRT_DISP_SQ
    disp_df = disp_df.join(disp_strt_sq)

    signed_disp = disp_df[io_tools.DISP] * disp_df[io_tools.DIRECTION]
    signed_disp.name = io_tools.SIGNED_DISP
    disp_df = disp_df.join(signed_disp)

    disp_array = disp_df[io_tools.DISP].to_numpy()
    total = len(disp_array)
    zero_count = total - np.count_nonzero(disp_array)
    pc = round(100 * zero_count / total, ndigits=2)

    print(f"{zero_count} of {total} are zero ({pc}%).")

    
    to_output_dir()

    #timestep histogram
    ordinal_histogram(
        disp_df, 
        io_tools.TIMESTEP, 
        title="Number of observations per timestep",
        xlabel="Change in time (seconds)", 
        ylabel="Number of observations",
        path=f"obs-per-timestep-bt{base_timestep}.svg"
    )

    for time in times:

        t_df = disp_df[disp_df[io_tools.TIMESTEP] == time]

        """
        #turning histogram
        ordinal_histogram(
            t_df, 
            io_tools.DIRECTION, 
            title=f"Turning frequency for time change {time} sec", 
            xtick_labels=['Reverse', 'Forward'],
            ylabel="Number of observations",
            path=f"turning-hist-t{time}.svg"
        )
        """

        #raw distance histogram
        numeric_histogram(
            t_df, 
            io_tools.DISP, 
            title=f"Absolute displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=f"abs-disp-hist-t{time}.svg"
        )

        #straight-line histogram
        numeric_histogram(
            t_df, 
            io_tools.STRT_DISP, 
            title=f"Displacement (straight line) histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=f"abs-disp-strt-hist-t{time}.svg"
        )
        
        """
        #signed distance histogram
        numeric_histogram(
            t_df, 
            io_tools.SIGNED_DISP, 
            title=f"Signed displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=f"sgn-disp-hist-t{time}.svg",
        )
        """

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
    """

    scatter_means(
        disp_df, 
        io_tools.TIMESTEP, 
        io_tools.DISP,
        title="Mean displacement vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"mean-disp-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        io_tools.TIMESTEP, 
        io_tools.DISP_SQ,
        title="Mean-squared displacement vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"mean-sq-disp-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        io_tools.TIMESTEP, 
        io_tools.STRT_DISP,
        title="displacement (straight line) vs. Time",
        xlabel="Change in time (seconds)",
        ylabel="Mean displacement (micrometers)",
        x_max=max_time, 
        path=f"disp-strt-scatter-bt{base_timestep}.svg"    
    )

    scatter_means(
        disp_df, 
        io_tools.TIMESTEP, 
        io_tools.STRT_DISP_SQ,
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
    t_df = disp_df[disp_df[io_tools.TIMESTEP] == time]

    ad_hist_files = []
    sd_hist_files = []
    strt_hist_files = []

    while time <= max_time:
        
        #raw distance histogram
        ad_name = f"abs-disp-hist-t{time}.jpg"
        ad_hist_files.append(ad_name)
        numeric_histogram(
            t_df, 
            io_tools.DISP, 
            title=f"Absolute displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts",
            path=ad_name, 
            xlims=[0, 100]
        )

        """
        #raw distance histogram
        sd_name = f"sgn-disp-hist-t{time}.jpg"
        sd_hist_files.append(sd_name)
        numeric_histogram(
            t_df, 
            io_tools.SIGNED_DISP, 
            title=f"Signed displacement histogram for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=sd_name, 
            xlims=[-100, 100]
        )
        """

        #raw distance histogram
        srt_name = f"strt-disp-hist-t{time}.jpg"
        strt_hist_files.append(srt_name)
        numeric_histogram(
            t_df, 
            io_tools.STRT_DISP, 
            title=f"Displacement histogram (straight line) for time change {time} sec", 
            xlabel="Displacment (micrometres)",
            ylabel="Counts", 
            path=srt_name, 
            xlims=[0, 100]
        )

        t_df = disp_df[disp_df[io_tools.TIMESTEP] == time]
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

    x_axis = df[io_tools.TIMESTEP].unique()
    x_axis.sort()

    y_axis_mean_disp = np.empty(len(x_axis))
    y_axis_mean_sq_disp = np.empty(len(x_axis))
    y_axis_mean_strt_disp = np.empty(len(x_axis))
    y_axis_mean_sq_strt_disp = np.empty(len(x_axis))
    for i, x in enumerate(x_axis):

        y_axis_mean_disp[i] = df[df[io_tools.TIMESTEP] == x][io_tools.DISP].mean()
        y_axis_mean_sq_disp[i] = df[df[io_tools.TIMESTEP] == x][io_tools.DISP_SQ].mean()
        y_axis_mean_strt_disp[i] = df[df[io_tools.TIMESTEP] == x][io_tools.STRT_DISP].mean()
        y_axis_mean_sq_strt_disp[i] = df[df[io_tools.TIMESTEP] == x][io_tools.STRT_DISP_SQ].mean()

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


    parser = argparse.ArgumentParser(description="Generate plots of cell displacement data.")
    parser.add_argument("infile", type=str, help="Path the cell displacement file.")
    parser.add_argument("base_timestep", type=int, help="The base timestep")

    parser.add_argument("--simulated", action="store_true")

    args = parser.parse_args()
       
    
    
    
    disp_df = pd.read_csv(args.infile, dtype=io_tools.DTYPE_DICT)
    



    
    
    if args.simulated:
        disp_df = io_tools.filter_simulated_displacement_data(disp_df)
    else:
        disp_df = io_tools.filter_displacement_data(disp_df, base_timestep=args.base_timestep)

    generate_all_plots(disp_df, args.base_timestep)

    disp_sq = disp_df[io_tools.DISP] ** 2
    disp_sq.name = io_tools.DISP_SQ
    disp_df = disp_df.join(disp_sq)

    disp_strt_sq = disp_df[io_tools.STRT_DISP] ** 2
    disp_strt_sq.name = io_tools.STRT_DISP_SQ
    disp_df = disp_df.join(disp_strt_sq)

    signed_disp = disp_df[io_tools.DISP] * disp_df[io_tools.DIRECTION]
    signed_disp.name = io_tools.SIGNED_DISP
    disp_df = disp_df.join(signed_disp)


    time_stats = time_statistics_df(disp_df)

    time_stats.to_csv(f"means{args.base_timestep}.csv")
