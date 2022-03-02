import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from scipy import stats

import stats_fitting
import io_tools
import folded_normal_mixture

skip_plotting = {"generic_gamma", "levy_leftskewed", "gamma", "half_generic_normal", "folded_cauchy", "levy", "inverse_gamma"}

FOLD_NORMAL_NAME = "folded_normal_mixture"

def plot_pdfs(ax, fits):

    cm = colormap.get_cmap('gist_ncar', len(fits['pdfs']))
    colors = iter(cm(range(len(fits['pdfs']) - len(skip_plotting))))

    x_hist, y_hist = fits['hist_data']
    

    bar_width = x_hist[1] - x_hist[0]

    ax.bar(x_hist, y_hist, bar_width, color='b', alpha=0.2)

    x = fits['x']
    

    names = []
    plot_elements = []
    for name in fits['pdfs']:
        

        if name in skip_plotting:
            continue

        y = fits['pdfs'][name]
        elem = ax.plot(x, y, color=next(colors))
        
        names.append(name)
        plot_elements.append(elem)

    ax.legend(names)

def get_distribution(name):
    if name in stats_fitting.candidate_distributions_stable:
        return stats_fitting.candidate_distributions_stable[name]
    elif name in stats_fitting.candidate_distributions_gamma:
        return stats_fitting.candidate_distributions_gamma[name]
    else: return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("fitfile", type=str, help="Json file storing fit data.")
    parser.add_argument("--datafile", type=str, help="Displacements data file", default=io_tools.DISPLACEMENTS_FILE_PATH)

    parser.add_argument("--outdir", type=str, help="Output directory. Defaults to current working directory")

    parser.add_argument("--simulated", action="store_true", help="set if the displacement data is simulated")

    args = parser.parse_args()

    df = pd.read_csv(args.datafile, dtype=io_tools.DTYPE_DICT)

    if args.simulated:
        df = io_tools.filter_simulated_displacement_data(df)
    else:
        df = io_tools.filter_displacement_data(df, base_timestep=10)

    disp_sq = df[io_tools.DISP] ** 2
    disp_sq.name = io_tools.DISP_SQ
    df = df.join(disp_sq)


    with open(args.fitfile) as f:
        fits = json.load(f)

    fts = dict()
    for key, val in fits.items():
        fts[int(key)] = val

    fits = fts

    #change to output directory if required
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        os.chdir(args.outdir)

    plot_data = dict()

    for t, data in stats_fitting.get_samples_over_time(df, io_tools.DISP, 1, 5):
        

        plot_data[t] = dict()

        x_hist, y_hist = stats_fitting.histogram(data, 50)

        bucket_width = x_hist[1] - x_hist[0]
        
        x_min = x_hist[0] - bucket_width / 2
        x_max = x_hist[-1] + bucket_width / 2

        x_axis = np.linspace(x_min+0.1, x_max - 0.1, num=100)


        plot_data[t]['hist_data'] = (x_hist, y_hist)
        plot_data[t]['x'] = x_axis
        plot_data[t]['pdfs'] = dict()
        plot_data[t]['sses'] = dict()
        for distribution, fit_data in fits[t].items():

            plot_data[t]['sses'][distribution] = fit_data['sse']

            if distribution == FOLD_NORMAL_NAME:
                p, m1, s1, m2, s2 = fit_data["params"]
                y_pdf = folded_normal_mixture.mix_pdf(x_axis, p, m1, s1, m2, s2)
                
            else:
                loc_p, scale_p = fit_data["params"][-2:]
                shape_ps = fit_data["params"][:-2]

                dist = get_distribution(distribution)

                y_pdf = dist.pdf(x_axis, loc=loc_p, scale=scale_p, *shape_ps)

            plot_data[t]['pdfs'][distribution] = y_pdf
            

    #plot the pdfs with the histogram for each time
    for t in plot_data.keys():


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)


        plot_pdfs(ax, plot_data[t])
        

        ax.set_xlabel("Displacement")
        ax.set_title(f"Displacement PDFs for t={t}s")
        plt.savefig(f"pdfs_t{t}.png")

    #plot sses
    


    ts = list(plot_data.keys())
    ts.sort()

    sses = dict()
    for t in sorted(plot_data.keys()):

        for dst, sse in plot_data[t]['sses'].items():

            if dst not in sses:
                sses[dst] = []

            sses[dst].append(sse)



    exit()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    cm = colormap.get_cmap('rainbow', len(sses.keys()) - len(skip_plotting))
    colors = iter(cm(range(len(sses.keys()))))

    names = []
    plot_elems = []
    for dst, sses in sses.items():
        

        print(dst, sses)

        if dst in skip_plotting:
            continue

        elem = ax.plot(ts, sses, color=next(colors), marker=".")

        names.append(dst)
        plot_elems.append(elem)


    ax.legend(names)

    ax.set_xlabel("Timestep (sec)")
    ax.set_ylabel("Sum of squared errors")

    plt.savefig("sses.png")
