from collections import defaultdict as dd
import pandas as pd
import math

import random

import io_tools
from stats_fitting import get_samples_over_time
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-13

def round_to_nearest_n(x, n):

    down = math.floor(x / n) * n
    up = math.ceil(x / n) * n

    #if they are equal choose randomly to avoid bias introduced by rounding 
    #choice. Can't use usual sign rule here since this can round to non-integers
    if abs((x - down) - (up - x)) < EPS:
        return random.choice([up, down])

    elif x - down > up - x:
        return up
    
    return down



def get_rounded_top_k(xs, k, rounder):

    freq = dd(int)

    for x in xs:

        freq[rounder(x)] += 1


    top_k = [(val, freq[val]) for i, val in zip(range(k), sorted(freq, key=freq.get, reverse=True))]


    return top_k


df = pd.read_csv(io_tools.DISPLACEMENTS_FILE_PATH, dtype=io_tools.DTYPE_DICT)
df = io_tools.filter_displacement_data(df, remove_zeros=False, base_timestep=10)


for t, sample in get_samples_over_time(df, io_tools.DISP, 10, max_time=120):

    print(f"For time {t}")

    for n in range(1, 11):

        rounded = np.array(get_rounded_top_k(sample, len(sample), lambda x : round_to_nearest_n(x, n)))


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.bar(rounded.T[0], rounded.T[1], color='b', alpha=0.5)
        ax.set_xlim(right=50)

        plt.savefig(f"t{t}_rounded_{n}.png")
        plt.close(fig)


    print()

