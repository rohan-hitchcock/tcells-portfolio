import random
import os
import sys
import itertools
from time import strftime
import json
from collections import defaultdict as dd

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import graphing
import config
import graph_tools as gt

from surfaces import Surface
from curves import PolyCurve
from vector import Vector
from tracks import Track, TrackPoint

from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy import interpolate

from min_bounding_rect import minBoundingRect as ext_min_bounding_box_2d

EPS = 1e-13

#*******************************************************************************
def random_normal(v, length=1):
    """Generates a normal vector of specified length in a random direction. If 
       v is the zero vector then the zero vector is returned.
       THIS IS FOR GENERATING TESTING DATA ONLY: no guaruntees are made about 
       the distribution (uniformity) of directions of the normal vectors"""
    if not v: 
        return v
    
    #find a non-zero entry in the tuple
    for k, v_k in enumerate(v):
        if v_k != 0:
            break
    
    #extract the section of vector before and after the non-zero entry
    v_data = v.get()
    v_prefix = v_data[:k]
    v_suffix = v_data[k+1:]

    #generate all but one of the normal vector entries randomly
    n_suffix = [random.uniform(-1, 1) for i in range(len(v_suffix))]
    n_prefix = [random.uniform(-1, 1) for i in range(len(v_prefix))]

    #last entry ensures the new vector is normal to v (the dot produc is zero)
    n_k = (-1 / v_k) * (sum(v_i * n_i for v_i, n_i in zip(v_suffix, n_suffix)) +
                        sum(v_i * n_i for v_i, n_i in zip(v_prefix, n_prefix)))

    return length * (Vector(n_prefix + [n_k, ] + n_suffix).unit())

#*******************************************************************************
def generate_polynomial(min_degree=1, max_degree=5, max_coeff=20, min_coeff=-20):
    """Generates a polynomial with random coefficients of at least degree 
       'min_degree' and at most degree 'max_degree'. Coefficents will be between 
       'max_coeff' and 'min_coeff'"""
    return Polynomial([random.uniform(min_coeff, max_coeff) 
                       for i in range(random.randint(min_degree, max_degree))])

#*******************************************************************************
def generate_sinusoid(poly_curve, num_points=1000, radius=0.125):
    """ Generates a sinusoid based on a curve. 
        Points are generated randomly along a normal vector at each point
        
        Args:
            poly_curve: an PolyCurve object
            num_points: the number of points to generate
            radius: the maximum distance along a normal vector a generated point
            will lie
        Yeilds:
            The next point in the sinusoid as a vector
        """

    d_poly_curve = poly_curve.deriv()

    for pt, tangent in zip(poly_curve.points(num_points), d_poly_curve.points(num_points)):
        pt = Vector(pt)
        tangent = Vector(tangent)

        normal = random_normal(tangent, radius)
        yield pt + normal

# HELPER FUNCTIONS *************************************************************
# These functions are not specifically related to plotting sinusoids
def create_filename(name="", default_extension="txt", word_sep="_"):
    time = strftime("%y%m%d-%H%M")

    prefix, sep, extension = name.rpartition(".")
    if not prefix and not sep:
        prefix = extension
        extension = default_extension
        sep = "."
    return time + word_sep + prefix + sep + extension

def pretty_poly(poly):
    return " + ".join(f"{c} x^{n}" for n, c in enumerate(poly))

# TESTING FUNCTIONS ************************************************************
# These functions test or demonstrate different parts of this program
def test_polynomial_sinusoid(savefile="", param_axis=2, num_plot_points=100, polys=None):
    """Generates and plots a sinusoid based on a randomly generated polynomial.
       By default a plot of the sinusoid and polynomial are displayed, 
       but if savefile is supplied it is written as an image to the file 
       specified by savefile"""

    poly_color = 'b'
    sin_color = 'r'
    
    if polys is None:
        polys = [generate_polynomial(), generate_polynomial()]
    sinusoid_curve = PolyCurve.graph(polys, param_axis=param_axis)

    print(sinusoid_curve)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graphing.mpl_scatter3d(ax, sinusoid_curve.points(num_plot_points), color=poly_color)
    graphing.mpl_scatter3d(ax, generate_sinusoid(sinusoid_curve), color=sin_color)
    
    fig.tight_layout()
    if savefile:
        plt.savefig(create_filename(savefile))
    else:
        plt.show()

#*******************************************************************************
def test_random_normal(num_to_generate, savefile="", dim=3):
    """Produces figures to show the distribution of anlges of generated normal
       vectors relative to each of the standard basis vectors. 'dim' specifies
       the dimension of the space."""
    
    def generate_normals(num_to_generate, dim=dim):
        for i in range(num_to_generate):  #pylint: disable=unused-variable

            v = Vector([random.random() for i in range(dim)])
            #to ensure these vectors are generated uniformly, only use ones 
            #which lie inside the unit sphere 
            #v cannot be zero so it has a well defined normal
            while v and v.length() <= 1:
                v = Vector([random.random() for i in range(dim)])

            yield random_normal(v)

    normals = list(generate_normals(num_to_generate))
    num_buckets = 10
    for i, vec in enumerate(Vector.standard_basis(dim)):
        fig, ax = plt.subplots() 
    
        graphing.mpl_histogram(ax, (vec.angle(n) for n in normals), num_buckets)

        ax.set_xlabel("Angle (rad)", size='x-large')
        ax.set_ylabel("Frequency", size='x-large')
        ax.set_title(f"Angle distribution in $x_{i}$ axis", size='x-large')

        fig.tight_layout()
        if savefile:
            plt.savefig(create_filename(str(i) + "_" + savefile))
        else:
            plt.show()

#*******************************************************************************
def test_fitting_generated(savefile="", num_sinusoid_points=10000, num_curve_points=100, show_sinusoid=False, param_axis=0, fit_degree=5):
    """Demonstrates the fitting process to a generated sinusoid by producing a
       plot showing the polynomial used to generate the sinusoid and a plot 
       showing the generated sinusoid. """

    #generate a sinusoid curve based on randomly generated polynomials
    sinusoid_curve = PolyCurve.graph([generate_polynomial(), generate_polynomial()], param_axis=param_axis)

    #we are working in three dimensions
    dim = 3

    #generate the sinusoid
    sinusoid = np.empty([num_sinusoid_points, dim])
    for i, pt in enumerate(generate_sinusoid(sinusoid_curve, num_points=num_sinusoid_points)):
        sinusoid[i] = np.array(tuple(pt))

    #produce the polynomials to fit the generated sinusoid data
    fit_curve = PolyCurve.least_squares_fit(sinusoid, param_index=param_axis, degrees=itertools.repeat(fit_degree))

    #output the results: 
    print(f"Source polynomial:\n\t{sinusoid_curve}"
          f"\n\nFitted polynomial:\n\t{fit_curve}")

    poly_color = 'b'
    sinusoid_color = 'r'
    fit_color = 'g'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if show_sinusoid:
        graphing.mpl_scatter3d(ax, sinusoid, color=sinusoid_color, 
                              marker=",", opacity=0.01)
    
    graphing.mpl_scatter3d(ax, sinusoid_curve.points(num_curve_points), color=poly_color)
    graphing.mpl_scatter3d(ax, fit_curve.points(num_curve_points), color=fit_color)

    fig.tight_layout()
    if savefile:
        plt.savefig(create_filename(savefile))
    else:
        plt.show()

#*******************************************************************************
def test_fitting_from_file(infile, param_axis=0, savefile="", sinusoid_as_points=False, surf_time_index=0):
    """ Demonstrates the fitting of a polynomial to a sinusoid saved to a file.
        The sinusoid is expected to be represented a triangulated surface. Only
        the vertices data is used to plot the polynomial. 

        Args:
            infile: the filename of a .npz file storing a triangulated surface
            param_axis: the axis about which the polynomial is parametrized
            savefile: save the resulting plot here
            sinusoid_as_points: set to True to plot only the vertices of the sinusoid
        Returns:
            None
    """

    #load sinusoid data from file
    surface = Surface.create_from_file(infile)
    surface_points = surface.points(surf_time_index)
    verts, faces, normals, values = surface.trisurf(surf_time_index)

    sinusoid_curve = PolyCurve.least_squares_fit(surface_points, param_index=param_axis, degrees=itertools.repeat(5))

    #setting up plotting axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(f"Sinusoid from {infile} about axis {param_axis}")

    #plot the sinusoid
    if sinusoid_as_points:
        graphing.mpl_scatter3d(ax, surface_points, marker=",", opacity=0.5, color='r')
    else:
        graphing.mpl_trisurface3d(ax, verts, faces, normals)

    graphing.mpl_scatter3d(ax, sinusoid_curve.points(100))

    if savefile:
        plt.savefig(create_filename(savefile))
    else:
        plt.show()

def check_surfaces():

    for t, spot_file, surf_files in config.data_iter(): #pylint: disable=unused-variable

        bad_surfs = []
        track_ids = set(str(t) for t in np.load(spot_file)['track_ids'])

        for surf in surf_files:
            corr_tracks = config.track_ids(surf)

            if not all(s in track_ids for s in corr_tracks):
                bad_surfs.append(surf)
            else:
                for tid in corr_tracks:
                    track_ids.remove(tid)

        current_file = os.path.split(spot_file)[0]

        if not bad_surfs:
            print(f"File '{current_file}' OK")

        else:
            print("---------------------------------------------------------")
            print(f"In file '{current_file}':")
            print("Unmatched surfaces:")
            for s in bad_surfs:
                print(f"\t{s}")
            print("\nRemaining track ids:")
            for tid in track_ids:
                print(f"\t{tid}")
            print("---------------------------------------------------------")

def plot_data(param_data=None, savefig=False, showfig=True):
    
    #time index for the surfaces
    surf_t = 0

    color_order = ['b', 'g', 'm', 'y', 'k']

    #allows for processes to run in the background while a figure is being displayed
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for time_step, spots_file, surface_files in config.data_iter(): #pylint: disable=unused-variable

        tracks = Track.from_spots_file(spots_file)

        for surface_file in surface_files:
            tids = config.track_ids(surface_file)
            surface = Surface.create_from_file(surface_file)

            verts, faces, normals, values = surface.trisurf(surf_t) #pylint: disable=unused-variable

            #plot the surface
            graphing.mpl_trisurface3d(ax, verts, faces, normals)

            #add the corresponding sinusoids
            for i, tid in enumerate(tids):
                graphing.mpl_scatter3d(ax, tracks[tid].points(), color=color_order[i])

            if (param_data is not None) and (param_data[surface_file] is not None):
                param_axis = param_data[surface_file]

                sinusoid = PolyCurve.least_squares_fit(surface.points(surf_t), param_axis)

                graphing.mpl_scatter3d(ax, sinusoid.points(100), color='r')


                for i, tid in enumerate(tids):
                    proj_track = project_track(sinusoid, tracks[tid])

                    graphing.mpl_scatter3d(ax, proj_track.points(),  color=color_order[i])

        if savefig:
            surface_name, sep, ext = surface_file.rpartition(".") #pylint: disable=unused-variable

            plt.savefig(surface_name + ".png")
        
        if showfig:
            plt.show()
            input("Press enter to continue...")

        ax.cla()

def check_surface_file(surface_file, tracks, save_fig=True):
    
    tids = config.track_ids(surface_file)
    surface = np.load(surface_file)

    verts, faces, normals, values = surface.trisurf(surface) #pylint: disable=unused-variable

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  

    #plot the surface
    graphing.mpl_trisurface3d(ax, verts, faces, normals)

    #add the corresponding sinusoids
    for tid in tids:
        graphing.mpl_scatter3d(ax, tracks[tid].points())

    plt.show()

    is_ok = str.lower(input("Surface OK? (y/n): ")) == "y"
    if is_ok and save_fig:
        plt.savefig(config.tid_separator.join(str(t) for t in tids) + ".png")
    
    plt.close()
    return is_ok

def choose_parameterisations(outfile, poly_degrees=[5, 5, 5]):

    #time index to sample surfaces from
    surface_time_index = 0

    param_axes = dict()

    #allows for processes to run in the background while a figure is being displayed
    plt.ion()

    fig = plt.figure()

    axes = []
    for i in range(0, 3):
        axes.append(fig.add_subplot(1, 3, i + 1, projection='3d'))

    for time_step, spots_file, surface_files in config.data_iter(): #pylint: disable=unused-variable

        for surface_file in surface_files:

            surface = Surface.create_from_file(surface_file)

            surf_points = surface.points(surface_time_index)
            verts, faces, normals, values = surface.trisurf(surface_time_index) #pylint: disable=unused-variable

            for param_index in range(0, 3):
                sinusoid = PolyCurve.least_squares_fit(surf_points, param_index, poly_degrees)

                graphing.mpl_scatter3d(axes[param_index], sinusoid.points(100))
                graphing.mpl_trisurface3d(axes[param_index], verts, faces, normals)

                axes[param_index].set_title(f"Axes {param_index}")

            plt.show()

            choice = input("Enter an axes (0, 1, 2): ")
            if choice not in ['0', '1', '2']:
                param_index = None
            else:
                param_index = int(choice)

            param_axes[surface_file] = param_index

            for ax in axes:
                ax.cla()

        with open(outfile, "w") as fp:
            json.dump(param_axes, fp, indent=4)

def max_length_as_vector(points):
    """ Finds the maximum length between points in a data set as a vector in the
        direction at which that length occurs.

        Args:
            An iterable collection of points of any dimension.

        Returns:
            A Vector object with magnitude the maximum length and direction the 
            direction in which this occurs.
    """

    pairs = itertools.combinations(points, 2)

    max_x, max_y = next(pairs)
    max_x = Vector(max_x)
    max_y = Vector(max_y)
    max_dist_sq = (max_x - max_y).length_sq()
    for x, y in pairs:

        x = Vector(x)
        y = Vector(y)

        dist_sq = (Vector(x) - Vector(y)).length_sq()

        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            max_x = x
            max_y = y
    return (max_x - max_y)

def find_optimal_transform(points):

    #it suffices to only consider the vertices of the convex hull enclosing 
    #the surface
    hull = ConvexHull(points)

    max_direction = max_length_as_vector((hull.points[i] for i in hull.vertices)).unit()

    x_axis = Vector(np.array([1, 0, 0]))

    #find a transformation rotating the max_direction to the x_axis 
    rotation_angle = x_axis.angle(max_direction)
    rotation_axis = (max_direction.cross(x_axis)).unit()
    xaxis_rotation = Rotation.from_rotvec((rotation_angle * rotation_axis).v)
    
    #rotate the hull points and project them onto the yz plane
    rot_hullpts_proj = np.array([xaxis_rotation.apply(hull.points[i])[1:] for i in hull.vertices])

    #now in 2d, find a convex hull for the projected points and repeat the process
    hull_2d = ConvexHull(rot_hullpts_proj)

    max_direction = max_length_as_vector((hull_2d.points[i] for i in hull_2d.vertices)).unit()

    #find a rotation for the max direction onto the (2d) x axis
    x_axis_2d = Vector(np.array([1, 0]))
    rotation_angle = -x_axis_2d.angle(max_direction)
    rotation_axis = x_axis

    yz_rotation = Rotation.from_rotvec((rotation_angle * rotation_axis).v)

    #return the total transformation 
    return yz_rotation * xaxis_rotation

def find_optimal_transform_new(points):

    #it suffices to only consider the vertices of the convex hull enclosing 
    #the surface
    hull = ConvexHull(points)

    max_direction = max_length_as_vector((hull.points[i] for i in hull.vertices)).unit()

    x_axis = Vector(np.array([1, 0, 0]))

    #find a transformation rotating the max_direction to the x_axis 
    rotation_angle = x_axis.angle(max_direction)
    rotation_axis = (max_direction.cross(x_axis)).unit()
    xaxis_rotation = Rotation.from_rotvec((rotation_angle * rotation_axis).v)
    
    #rotate the hull points and project them onto the yz plane
    rot_hullpts_proj = np.array([xaxis_rotation.apply(hull.points[i])[1:] for i in hull.vertices])

    #now in 2d, find a convex hull for the projected points and repeat the process
    hull_2d = ConvexHull(rot_hullpts_proj)

    minbox_corners = min_area_bounding_box_2d(hull_2d)

    #aligning the y axis with either the long or the short axis of the box will do
    new_xaxis_2d = Vector(minbox_corners[0] - minbox_corners[1]).unit()

    #find a rotation for the max direction onto the (2d) x axis
    xaxis_2d = Vector(np.array([1, 0]))
    rotation_angle = -xaxis_2d.angle(new_xaxis_2d)
    #rotate about the original axis
    rotation_axis = x_axis

    yz_rotation = Rotation.from_rotvec((rotation_angle * rotation_axis).v)

    #return the total transformation 
    return yz_rotation * xaxis_rotation

def min_area_bounding_box_2d(hull):
    """ Finds the minumum-area bounding box for a convex hull in 2 dimensions.
    
        Args:
            hull: A 2-dimensional ConvexHull object
            
        Returns:
            An numpy array of shape (4, 2) representing the coordinates of 
            four corners of the minimum-area bounding box of hull. Coordinates
            are listed clockwise.        
    """

    hull_pts = np.empty((len(hull.vertices) + 1, 2))
    for i, point_index in enumerate(hull.vertices):
        hull_pts[i] = hull.points[point_index]
    
    hull_pts[-1] = hull_pts[0]

    #pylint: disable=unused-variable
    angle, area, width, height, centre, corner_pts = ext_min_bounding_box_2d(hull_pts)

    return corner_pts

def create_sinusoid_plots(image_format='svg', plot_time=0):
    """Creates and saves a plot of all sinusoid surfaces."""
    #pylint: disable=unused-variable
    for time_step, spots_file, surface_files in config.data_iter(): 

        for surface_file in surface_files:

            surface = Surface.create_from_file(surface_file)

            #pylint: disable=unused-variable
            verts, faces, normals, values = surface.trisurf(plot_time)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            graphing.mpl_trisurface3d(ax, verts, faces, normals)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            fig.tight_layout()

            #pylint: disable=unused-variable
            surf_name, ext = os.path.splitext(surface_file)

            image_file = surf_name + "." + image_format

            plt.savefig(image_file, format=image_format)

            plt.close(fig)

# MAIN *************************************************************************
if __name__ == "__main__":
    """
    tracks = Track.from_spots_file(config.test_spots)
    
    AV_SINUSOID_DIAMETER = 6

    for track in tracks.values():
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        nng = gt.nearest_neighbor_graph(track.points())

        simplified_nng_0 = gt.simplify(nng, cluster_distance=AV_SINUSOID_DIAMETER)

        simp_points = gt.simplify_points(track.points(), cluster_distance=AV_SINUSOID_DIAMETER)
        simplified_nng = gt.nearest_neighbor_graph(simp_points)
    
        graphing.mpl_r3graph(ax, nng, node_color='k', edges=False)
        graphing.mpl_r3graph(ax, simplified_nng, nodes=False, edge_color='g')
        graphing.mpl_r3graph(ax, simplified_nng_0, nodes=False, edge_color='b')
        graphing.mpl_set_labels(ax)

        plt.show()
        plt.close(fig)

    """

    os.chdir(config.working_dir)
    os.chdir(config.input_dir)

    base_timesteps = dd(list)
    for timestep, spots_file, surf_files in config.data_iter():
        base_timesteps[timestep].append("/".join(config.get_experiment(spots_file)))

    path = "/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/data_processed/NovDec2019_corrected/base_timesteps.json"

    with open(path, "w") as f:
        json.dump(base_timesteps, f, indent=4)

        