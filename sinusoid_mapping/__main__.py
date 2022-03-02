import argparse
import os
import time 

from sinusoid_mapping import process
from sinusoid_mapping import config


def insert_time_tag(path):
    head, tail = os.path.split(path)
    return os.path.join(head, time.strftime("%y%m%d-%H%M") + tail)

def ensure_exists(path):
    head, tail = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)


parser = argparse.ArgumentParser(
    description="This module fits curves to sections of sinusoid and processes track data.")

parser.add_argument("outfile", help="The name of the output file. If a relative path is given this will be relative to the working directory specified in config.py. If --plots is used this should be a directory.")

parser.add_argument("--plots", action="store_true", help="Writes plots (images) of tracks and their sinusoids.")
parser.add_argument("--include_end_proj", action="store_true", help="Include track projections which map exactly to the end points of the sinusoid.")

parser.add_argument("--simulated_data", type=str, default="", help="File path to simulated cell movement data")

parser.add_argument("--max_track_points", type=int, default=None, help="The maximum length of a track to use.")

args = parser.parse_args()


if args.plots:
    os.chdir(config.working_dir)
    os.chdir(config.input_dir)


    if not os.path.isdir(args.outfile):
        os.makedirs(args.outfile)

    process.produce_plots(path=args.outfile, tag="with_surface", show_surf=True)
    process.produce_plots(path=args.outfile, tag="without_surface")

elif args.simulated_data:

    head, tail = os.path.split(args.simulated_data)
    name, ext = os.path.splitext(tail)

    df = process.calculate_simulated_displacements(args.simulated_data, "000000", name, exclude_end_projections=not args.include_end_proj, max_track_points=args.max_track_points)

    df.to_csv(args.outfile)
    
else:
    os.chdir(config.working_dir)
    os.chdir(config.input_dir)

    disp_data = process.calculate_displacements(config.data_iter(), exclude_end_projections=not args.include_end_proj, max_track_points=args.max_track_points)

    print(disp_data)

    file_path = insert_time_tag(args.outfile)
    ensure_exists(file_path)

    disp_data.to_csv(file_path)
    

