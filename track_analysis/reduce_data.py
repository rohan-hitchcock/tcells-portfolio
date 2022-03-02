import argparse
import numpy as np
import pandas as pd

import io_tools

BASE_TIMESTEP = 10
MAX_TIMESTEP = 120

parser = argparse.ArgumentParser(description="Reduce displacement data for remote processing.")

parser.add_argument("disp_file", help="displacements file")
parser.add_argument("outfile", help="output file")

args = parser.parse_args()

df = pd.read_csv(args.disp_file, dtype=io_tools.DTYPE_DICT)

df = io_tools.filter_displacement_data(df, remove_zeros=False, base_timestep=BASE_TIMESTEP)

df = df[df[io_tools.TIMESTEP] <= MAX_TIMESTEP]

df.to_csv(args.outfile)
