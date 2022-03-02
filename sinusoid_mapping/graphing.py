import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#*******************************************************************************
def mpl_scatter3d(ax, points, color='b', marker=".", opacity=1):
    """Plots points as a three dimensional scatter plot on a matplotlib axis
    
        Args:
            ax: a matplotlib axis object
            points: an array-like containg three dimensional points
    """
    points = np.array(points, copy=False)
    ax.scatter(points[:,0], points[:,1], points[:,2], c=color, marker=marker, alpha=opacity)
    
#*******************************************************************************
def mpl_trisurface3d(ax, verts, faces, normals, color='r', opacity=0.4):
    """Plots a triangulated surfaces in three dimensions
        
    Args:
        ax: a 3d matplotlib axis to plot the surface on
        verts: an array of coordinates of the 3d mesh of the surface
        faces: an array of groups of 3 indices of the `verts` array, which 
               define the triangular faces of the surface
        normals: an array of normal 3-vectors, for each vertex on the mesh
    Returns:
        None (mutates ax)"""
    ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2], color=color, alpha=opacity)

#*******************************************************************************
def mpl_line3d(ax, points, color='b', opacity=1, linewidth=1):
    """Plots points as a three dimensional line plot on a matplotlib axis
    
        Args:
            ax: a matplotlib axis object
            points: an array-like containg three dimensional points
    """
    #ensure it is a numpy array so we can use slicing
    points = np.array(points, copy=False)
    ax.plot(
        points[:,0], points[:,1], points[:,2], 
        linestyle='-', 
        c=color, 
        alpha=opacity, 
        linewidth=linewidth
    )

#*******************************************************************************
def mpl_set_labels(ax, title="", xaxis="X", yaxis="Y", zaxis="Z"):
    """ Updates the lables of a 3d matplotlib axis.
        
        ax: the matplotlib axis to update
        title: the title of this axis
        xaxis: the label for the x axis for this axis
        yaxis: the label for the y axis for this axis
        zaxis: the label for the z axis for this axis
    """

    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_zlabel(zaxis)

def mpl_r3graph(ax, graph, nodes=True, edges=True, node_color='r', edge_color='b'):
    """ Plots a graph which has nodes which are points in three-dimensional 
        space
        
        ax: a 3d matplotlib axis to plot on
        graph: a networkx graph with 3d-points as nodes
    """
    if nodes:
        mpl_scatter3d(ax, list(graph.nodes), color=node_color)

    if edges:
        for e in graph.edges:
            mpl_line3d(ax, e, color=edge_color)

#*******************************************************************************
def mpl_histogram(ax, data_iter, num_buckets, colour='b', opacity=0.6):
    #get the maximum and minimum of the data to calculate the bucket size
    data = list(sorted(data_iter))
    min_val = data[0]
    max_val = data[-1]
    
    bucket_size = float(max_val - min_val) / num_buckets
    
    #count the number of values in each bucket
    curr_key = min_val + bucket_size
    frequency = {curr_key: 0}

    #put the values into buckets in order
    for val in data:

        #the current value should be placed in a higher bucket
        if curr_key < val:
            while (curr_key < val):
                curr_key += bucket_size
            
            #ensure empty buckets are detected
            frequency[curr_key] = 0
        
        #count the current value
        frequency[curr_key] += 1
    
    xdata = list(sorted(frequency.keys()))
    counts = [frequency[key] for key in xdata]
    ax.bar(xdata, counts, width=(0.9*bucket_size), alpha=opacity, color=colour)

#*******************************************************************************
if __name__ == "__main__":
    f = lambda t : t ** 2
    g = lambda t : t ** 2 - t + 1
    pts = [(x, f(x), g(x)) for x in np.linspace(-1, 1)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    mpl_set_labels(ax)
    fig.tight_layout()

    mpl_scatter3d(ax, pts)

    plt.show()
    