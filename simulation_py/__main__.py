import argparse
import time
from graph_processing import mxf_tools as mxf
from simulation import simulate
from simulation import reporting

#default argument values
D_LOBULE_FILE = "./tissue-graphs/spline_weighted.mxf"
D_OUTPUT_DIR = "."

parser = argparse.ArgumentParser(
    description="This program simulates T cells moving in liver lobules."
)

#required arguments (in order)
parser.add_argument("sim_name", type=str, help="The name of this simulation")
parser.add_argument("num_cells", type=int, help="Number of T cells to simulate.")
parser.add_argument("iterations", type=int, help="Number of iterations of simulation.")
parser.add_argument("movement", type=str, help="The cell movement algorithm to use.")

#optional arguments
parser.add_argument("--lobule", type=str, default=D_LOBULE_FILE, help="Path to the lobule file to use.")
parser.add_argument("--output_dir", type=str, default=D_OUTPUT_DIR, help="Path to directory to store results.")
parser.add_argument("--positions", "-P", action='store_true', help="Set to record cell positons.")
parser.add_argument("--coverage", "-C", action="store_true", help="Set to record cell coverage.")
parser.add_argument("--metadata", "-M", action="store_true", help="Set tp record simulation metadata")

parser.add_argument("--params", "-p", nargs="*", type=float, help="A list of parameters for the movement type. The number of values and their meaning depends on the selected movement.")

args = parser.parse_args()

#include current time in sim name to avoid conflicts
full_sim_name = time.strftime("%y%m%d-%H%M") + args.sim_name

tissue = mxf.read(args.lobule)
cells = simulate.initialise(tissue, args.num_cells)

reporter = reporting.SimReporter("./", full_sim_name)

if args.coverage:
    reporter.set_report_coverage(tissue.number_of_nodes())

if args.positions:
    reporter.set_report_positions()

if args.metadata:
    reporter.set_report_metadata({
        "num_cells": args.num_cells, 
        "total_iterations": args.iterations, 
        "movement": args.movement,
        "params": args.params})

if args.movement.lower() == "brownian":

    if len(args.params) != 1:
        print("Incorrect number of parameters for brownian movement type (require 1)")
        exit()

    d  = args.params[0]
    mover = lambda g, c: simulate.move_brownian(g, c, d)

elif args.movement.lower() == "simple_forward":

    try:
        t_p, d  = args.params[0]
    except IndexError:
        print("Incorrect number of parameters for simple_forward movement type (require 2)")

    mover = lambda g, c: simulate.move_simple_forward(g, c, d, t_p)

elif args.movement.lower() == "normal_mixture":

    try:
        t_p, m_p, m1, s1, m2, s2 = args.params
    except ValueError:
        print("Incorrect number of parameters for normal_mixture movement type (require 6)")
        exit()


    mover = lambda g, c: simulate.move_normal_mixture(g, c, t_p, m_p, m1, s1, m2, s2)

elif args.movement.lower() == "normal":
    try:
        t_p, m, s = args.params
    except ValueError:
        print("Incorrect number of parameters for normal movement type (require 3)")
        exit()

    mover = lambda g, c : simulate.move_normal(g, c, t_p, m, s)

# INSERT OPTIONS FOR OTHER MOVEMENT TYPES HERE

else:
    print("movement style not recognised")
    exit()


with reporter as reporter:
    for t in range(args.iterations):

        for i, cell in enumerate(cells):

            cells[i], path = mover(tissue, cell)
            
            reporter.update_move(t, i, cells[i], path)

        reporter.end_iteration(t)
