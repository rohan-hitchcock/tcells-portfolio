from collections import defaultdict as dd
import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()


parser.add_argument("fitfile")

parser.add_argument("--outdir", type=str, help="Directory to write output files.")

parser.add_argument("--simulated", action="store_true")


args = parser.parse_args()


with open(args.fitfile) as f:
    fits = json.load(f)

#change to output directory if required
if args.outdir:
    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)


if args.simulated:
    ts = np.arange(start=1, stop=13, step=1)
else:
    ts = np.arange(start=10, stop=121, step=10)

params = dd(list)

for t in ts:

    t_key = str(t)

    for dist_name in fits[t_key]:

        params[dist_name].append(fits[t_key][dist_name]['params'])



for dist in params:

    ps = np.array(params[dist])

    for i in range(len(ps.T)):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        name = f"{dist}_param_{i}"

        ax.set_title(name)
        ax.set_xlabel("time")
        ax.set_ylabel("parameter val")

        ax.plot(ts, ps.T[i])
        

        plt.savefig(name + ".png")
        plt.close(fig)
