# T-Cell modelling project

This repository collects selected code from my time working on a 
cross-disciplinary project at the Peter Doherty Institute and the University 
of Melbourne. The goal of this project was to determine a stochastic model for 
the motion of a particular population of T-cells residing in the liver. This 
project was supervised by Dr Lynette Beattie (Dept. Microbiology and 
Immunology), Professor Jonathan Manton 
(Dept. Electrical and Electronic Engineering) and Professor William Heath 
(Dept. Microbiology and Immunology), all at The University of Melbourne. 
Some data (not included in this repository) and advice was also provided by
Professor Stefan Hoehme at Leipzig University. 

Not all of the code written as part of this project has been included in this 
repository, and no data has been included. 

## Components of this repository
- `sinusoid_mapping`: this module is concerned with approximating the sinusoids 
using a connected network of curves, and extracting the displacements of cells 
moving along these curves (i.e. looking at the distance along the curve between 
to points rather than the straight line distance). It takes microscope imaging 
data after it has been analysed by Imaris (a program that works directly with 
the output of the microscope). Components can be used individually, but the 
main entry point is `sinusoid_mapping.process`
- `track_analysis`: consists of code used to analyse the output of the `sinusoid_mapping`
module. A major goal is to understand the distribution of the displacements of 
a cell moving along the sinusoids. This module contains many attempts to estimate 
the distribution of these displacements, as well as code for visualising this 
data. 
- `simulation_c`: can be compiled into a simulation of cells moving through a liver. 
This requires a graphical (i.e. nodes and edges) representation of a liver which is 
not provided. 
- `simulation_py`: achieves the same goal as `simulation_c`, but it is written 
in python. It is less performant but easier to change and is mostly used for 
prototyping. 