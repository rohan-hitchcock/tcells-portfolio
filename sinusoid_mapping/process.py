import heapq
import os
from recordclass import recordclass
import itertools
import time
import math

import numpy as np
import pandas as pd
import networkx as nx
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sinusoid_mapping import config
from sinusoid_mapping import graph_tools
from sinusoid_mapping import graphing 
from sinusoid_mapping import curves
from sinusoid_mapping import surfaces
from sinusoid_mapping.tracks import Track

EPS = 1e-13

AV_SINUSOID_DIAMETER_MICROMETER = 6.0

#the names of attributes used when outputing data
DISPLACEMENT_ATTR = 'curve_displacement (micro-m)'
DISP_ERROR_ATTR = 'abs_displacement_numeric_error'
TIMESTEP_ATTR = 'time_step'
TID_ATTR = 'track_name'
EXPNAME_ATTR = 'experiment'
DATE_ATTR = 'date'
TRACK_LENGTH_ATTR = 'num_points'
TRACK_DURATION_ATTR = 'duration (sec)'
RELDIR_ATTR = 'reldir'
STRT_DISP = 'straight displacement (micro-m)'


#******************************************************************************
def simplify_points(points, cluster_distance):
    """ Simplifies points by finding the minimum number of distjoint sets which 
        cover points (clusters) such that any two points are in the same cluster 
        are at most cluster_distance apart. The simplified points are the center 
        of mass of each of these clusters.

        Args:
            points: a finite iterable of point-like things.
            cluster_distance: the maximum Euclidean distance of two points in
            the same cluster

        Yields:
            The center of mass of each cluster
    """

    #represents a cluster in the priority queue, priority is -len(cluster)
    PCluster = recordclass("PCluster", ["priority", "cluster"])

    #updates the cluster priority queue
    def update_pqueue(pqueue, max_cluster):

        insert_at = 0
        for i, p_cluster in enumerate(pqueue):

            p_cluster.cluster -= max_cluster
            p_cluster.priority = -len(p_cluster.cluster)

            if p_cluster.priority != 0:

                if i != insert_at:
                    pqueue[insert_at] = p_cluster
                
                insert_at += 1
        #delete empty clusters and restore heap condition
        del pqueue[insert_at:]
        heapq.heapify(pqueue)
        
    #square this now to save lots of square roots later
    cluster_distance = cluster_distance ** 2
    
    #build a priority queue of clusters around each point
    cluster_pqueue = []
    for p in points:

        #construct a cluster of points within cluster_distance of p
        cluster = set()
        for q in points:
            if sum((x - y) ** 2 for x, y in zip(p, q)) < cluster_distance:
                cluster.add(tuple(q))
        
        #add cluster to priority queue. The heapq implementation is a min-heap, 
        #so set the priority as the negation of length
        cluster_pqueue.append(PCluster(priority=-len(cluster), cluster=cluster))
    
    heapq.heapify(cluster_pqueue)

    #produce the center of mass of each cluster, in cluster size order, 
    #removing points as we go
    dim = len(next(iter(cluster_pqueue[0].cluster)))
    simplified_points = np.empty((len(cluster_pqueue), dim))
    i = 0
    while cluster_pqueue:
        
        #find the center of mass of the largest cluster and add it as a node
        c_max = heapq.heappop(cluster_pqueue).cluster
        simplified_points[i] = centre_of_mass(c_max)
        i += 1
        update_pqueue(cluster_pqueue, c_max)

    simplified_points.resize((i, dim), refcheck=False)
    return simplified_points

#******************************************************************************
def centre_of_mass(points):
    """ Returns the centre of mass of a collection of points as a numpy array.

        Args:
            points: a finite, non-empty iterable of points
        
        Returns:
            The centre of mass point as a numpy array
    """
    return sum(np.array(p, copy=False) for p in points) / len(points)

#******************************************************************************
# Function for calculating track displacements 
def calculate_displacements(data, exclude_end_projections=False, max_track_points=None):

    table = {
        DATE_ATTR: [],
        EXPNAME_ATTR: [],
        TID_ATTR: [],
        TIMESTEP_ATTR: [],
        TRACK_LENGTH_ATTR: [],
        TRACK_DURATION_ATTR: [],
        DISPLACEMENT_ATTR: [],
        DISP_ERROR_ATTR: [], 
        RELDIR_ATTR: [], 
        STRT_DISP: []
    }

    #pylint: disable=unused-variable
    for time_step, spots_file, surf_files in data:
        
        tracks = Track.from_spots_file(spots_file)
        
        datestr, experiment = config.get_experiment(spots_file)

        print(f"processing {datestr}/{experiment}")
        num_tracks = len(tracks)
        counter = itertools.count()
        next(counter)
        
        for tid, track in tracks.items():  

            c = next(counter)
            if c % 5 == 0:
                print(f"{c}/{num_tracks} tracks for this file.")

            
            compute_track_values(table, tid, track, time_step, datestr, experiment, exclude_end_projections, max_track_points)

    return pd.DataFrame(table)
    
def calculate_simulated_displacements(datafile, datestr, experiment, exclude_end_projections=False, max_track_points=None):
    
    table = {
        DATE_ATTR: [],
        EXPNAME_ATTR: [],
        TID_ATTR: [],
        TIMESTEP_ATTR: [],
        TRACK_LENGTH_ATTR: [],
        TRACK_DURATION_ATTR: [],
        DISPLACEMENT_ATTR: [],
        DISP_ERROR_ATTR: [], 
        RELDIR_ATTR: [], 
        STRT_DISP: []
    }

    tracks = Track.from_csv(datafile, tid="cell_id", time="iteration")

    num_tracks = len(tracks)
    print(num_tracks)
    counter = itertools.count() 
    next(counter)

    #use unit timestep, so time is measured in number of iterations/frames
    time_step = 1

    for tid, track in tracks.items():
        c = next(counter)

        #if c % 5 == 0:
        print(f"{c}/{num_tracks} tracks for this file.")

        
        compute_track_values(table, tid, track, time_step, datestr, experiment, exclude_end_projections, max_track_points)
    
    return pd.DataFrame(table)
    

def compute_track_values(table, tid, track, time_step, datestr, experiment, exclude_end_projections, max_track_points=None):
    sinusoid_points = simplify_points(
        track.points(), 
        AV_SINUSOID_DIAMETER_MICROMETER
    )
    nng = graph_tools.nearest_neighbor_graph(sinusoid_points)

    if len(nng.nodes) < 2:
        print(f"WARNING: Skipping track {tid} in {datestr}/{experiment}: "
                f"Nearest neighbor graph has {len(nng.nodes)} node.")
        return

    if (not nx.is_tree(nng)):
        print(f"WARNING: Skipping track {tid} in {datestr}/{experiment}: "
                "Nearest neighbor graph is not a tree.")
        return

    sinusoid = curves.GraphCurve.create_from_nng(nng, interpolate_spline)

    if (not nx.is_tree(sinusoid.curve_graph.to_undirected(reciprocal=True))):
        print(f"WARNING: Skipping track {tid} in {datestr}/{experiment}: Sinusoid graph is not a tree.")
        return

    proj_track = project_track(sinusoid, track, exclude_end_projections=exclude_end_projections)

    track_length = len(proj_track)

    #choose a random continuous subtrack if we have a limit on the number of points
    if track_length > max_track_points:
        start = np.random.randint(track_length - max_track_points)
        end = start + max_track_points
        proj_track = proj_track.get_slice(start, end)

    track_time_duration = proj_track.duration() * time_step


    for delta in range(1, proj_track.duration() + 1):
        
        time_delta = delta * time_step

        #stores the arrival directions at each point
        arrive_dirs = dict()

        for p1, p2 in proj_track.pairwise(delta):
            
            #coordinate in 3d and parameter on curve of p1
            p1_coord, p1_param = p1.position
            p2_coord, p2_param = p2.position

            displacement, disp_err = sinusoid.length(p1_param, p2_param)
            

            dir_p1leave, dir_p2arrive = sinusoid.directions(p1_param, p2_param)
            

            if dir_p2arrive != 0:
                arrive_dirs[p2] = dir_p2arrive
            
            #this happens if the path starts or ends at a curve end
            else:
                pass
                #print(f"WARNING: A local direction in track {tid} in {datestr}/{experiment} equals zero.")

            dir_p1arrive = arrive_dirs.get(p1)

            if dir_p1arrive is not None:
                #say this is a 'positive' displacement 
                reldir = 1 if dir_p1arrive == dir_p1leave else -1
            #relative direction is not definable (eg. we don't know arival direction)
            else:
                reldir = np.nan
            

            straight_line_displacement = math.sqrt(sum((x - y) **2 for x, y in zip(p1_coord, p2_coord)))
            
            table[DATE_ATTR].append(datestr)
            table[EXPNAME_ATTR].append(experiment)
            table[TID_ATTR].append(tid)
            table[DISPLACEMENT_ATTR].append(displacement)
            table[DISP_ERROR_ATTR].append(disp_err)
            table[TIMESTEP_ATTR].append(time_delta)
            table[TRACK_LENGTH_ATTR].append(track_length)
            table[TRACK_DURATION_ATTR].append(track_time_duration)
            table[RELDIR_ATTR].append(reldir)
            table[STRT_DISP].append(straight_line_displacement)

def project_track(gc, track, exclude_end_projections=False):
    
    proj_track = Track()

    for t, p in track:
        
        proj_point_param = gc.closest_point_parameter(p)


        if (not exclude_end_projections) or (not gc.is_end_point(proj_point_param)):
            proj_track.insert(t, (tuple(p), proj_point_param))

    return proj_track

def interpolate_spline(points, dim=3):
    if len(points) <= dim:
        return interpolate.splprep(points.transpose(), k=dim-2)[0]
    return interpolate.splprep(points.transpose())[0]

#*******************************************************************************
#Functions related to producing plots of tracks, surfaces and sinusoids

def produce_plots(path="", tag="", show_track=True, show_sinusoid=True, show_surf=False, show_nng=False):

    #pylint:disable=unused-variable
    for time_step, spots_file, surf_files in config.data_iter():

        tracks = Track.from_spots_file(spots_file)
        datestr, experiment = config.get_experiment(spots_file)

        print(f"processing {datestr}/{experiment}")

        surfs = dict()
        for surf_file in surf_files:
            surfs.update({tid: surf_file for tid in config.track_ids(surf_file)})

        for tid, track in tracks.items():
            
            
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_ylabel("z (um)")

            if show_track:
                graphing.mpl_scatter3d(ax, track.points(), color='k')

            if show_sinusoid or show_nng:
                sinusoid_points = simplify_points(
                    track.points(), 
                    AV_SINUSOID_DIAMETER_MICROMETER
                )

                nng = graph_tools.nearest_neighbor_graph(sinusoid_points)

                if show_nng:
                    graphing.mpl_r3graph(ax, nng, node_color='b', edge_color='b')

                if show_sinusoid:
                    sinusoid = curves.GraphCurve.create_from_nng(nng, interpolate_spline)

                    for branch in sinusoid.branches():
                        start, stop = branch.domain()
                        line = np.linspace(start, stop, num=200)

        
                        pts = [branch(x) for x in line]
                        graphing.mpl_line3d(ax, pts, color='g')

            if show_surf and tid in surfs:
                surface = surfaces.Surface.create_from_file(surfs[tid])

                verts, faces, normals, values = surface.trisurf(0)

                graphing.mpl_trisurface3d(ax, verts, faces, normals, color='r')
            
            image_filepath = image_filename(tag, path, datestr, experiment, str(tid))

            image_path, image_name = os.path.split(image_filepath)

            os.makedirs(image_path, exist_ok=True)

            plt.savefig(image_filepath)
            plt.close(fig)

def image_filename(tag, pth, datestr, experiment, tid, ext="svg"):
    return os.path.join(pth, datestr, experiment, tid + "_" + tag) + "." + ext

def show_plots(spots_tid_dict):

    for spots_file, tids in spots_tid_dict.items():

        tracks = Track.from_spots_file(spots_file)

        for tid, track in tracks.items():
            
            if tid not in tids:
                continue

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_ylabel("z (um)")

            sinusoid_points = simplify_points(
                    track.points(), 
                    AV_SINUSOID_DIAMETER_MICROMETER
                )

            nng = graph_tools.nearest_neighbor_graph(sinusoid_points)

            
            sinusoid = curves.GraphCurve.create_from_nng(nng, interpolate_spline)

            for branch in sinusoid.branches():
                start, stop = branch.domain()
                line = np.linspace(start, stop, num=200)

        
                pts = [branch(x) for x in line]
                graphing.mpl_line3d(ax, pts, color='g')
            
            graphing.mpl_scatter3d(ax, track.points(), color='k')

            plt.show()
            plt.close(fig)

#******************************************************************************
if __name__ == "__main__":
    
    os.chdir("/home/rohan/Dropbox/doherty/experiments/11-mapping-sinusoids/data/NovDec2019_corrected")
    
    interesting_tracks = {
        "./191206/mouse1_6_30sec/Spots 1.npz": [1000001164, 1000000345], 
        "./191206/mouse1_4_30sec/Spots 1.npz": [1000014899], 
        "./191205/mouse1_1_drift_corrected/Spots 1.npz": [1000059656], 
        "./191203/mouse1_1_drift_corrected/Spots 1.npz": [1000030933]
    }

    show_plots(interesting_tracks)
    