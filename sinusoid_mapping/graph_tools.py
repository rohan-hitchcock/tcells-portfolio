import heapq
import itertools
from recordclass import recordclass

import networkx as nx
import numpy as np

#******************************************************************************
def nearest_neighbor_graph(points):
    """ Creates the connected graph of points where the total Euclidean distance 
        between connected points is minimal.
        
        Args:
            A finite iterable of points.

        Returns: 
            A networkx graph as above.
    """
    #represents a node in the priority queue of unconnected nodes
    UNode = recordclass("UnconnectedNode", ["distance", "node", "closest"])

    #function for updating elements of the priority queue
    def update_pqueue(pqueue, new_node):

        for u_node in pqueue:
            dist_to_new = sum((x - y) ** 2 for x, y in zip(u_node.node, new_node))

            if dist_to_new < u_node.distance:
                u_node.distance = dist_to_new
                u_node.closest = new_node   

        heapq.heapify(pqueue)

    #create a graph of the points, converted to tuples so they are hashable
    nn_graph = nx.Graph()
    nn_graph.add_nodes_from(tuple(p) for p in points)
    
    #create a priority queue of unconnected nodes sorted by distance from 
    #the connected nodes in the graph
    unconnected_nodes = [UNode(distance=float("inf"), node=n, closest=None)
                        for n in nn_graph.nodes]   

    #if there are no points, return an empty graph
    if not unconnected_nodes:
        return nn_graph
    
    update_pqueue(unconnected_nodes, unconnected_nodes.pop().node)

    #connect all nodes in the graph
    while unconnected_nodes:
        
        #connect the closest unconnected node and update the priority queue
        new_node = heapq.heappop(unconnected_nodes)
        nn_graph.add_edge(new_node.node, new_node.closest)
        update_pqueue(unconnected_nodes, new_node.node)

    return nn_graph

#******************************************************************************
def longest_path(tree):
    """ Returns a longest path of some connected component of tree. If tree has
        more than one connected component, or if more than one longest path 
        exists the path returned will be arbitrary. It will be a path of 
        maximal length in its connected component. 
        
        Args:
            tree: A networkx graph which is a tree (no cycles), which has nodes
            that can be converted to flat numpy arrays
        
        Returns:
            A list of nodes which is the longest path in tree.
    """
    #choose first source arbitrarily
    src = next(iter(tree.nodes))

    #perform a depth first search and record the distance of each node to src
    dists = {src: 0}
    for tail, head in nx.dfs_edges(tree, src):
        dists[head] = 1 + dists[tail]
    
    #a longest path in this tree will start at the node farthest from src
    longest_path_start = max(dists, key=dists.get)

    #perform a second dfs, recording distances and predecessors
    dists = {longest_path_start: 0}
    preds = {}
    for tail, head in nx.dfs_edges(tree, longest_path_start):
        dists[head] = 1 + dists[tail]
        preds[head] = tail
    
    #the other end of this longest path will be the node farthest from the start
    longest_path_end = max(dists, key=dists.get)


    dim = len(longest_path_end)
    path_length = dists[longest_path_end] + 1

    path = np.empty((path_length, dim))
    i = 0

    #reconstruct the path using the predecessors dict
    path[i] = np.array(longest_path_end)
    curr_node = longest_path_end
    next_node = preds.get(curr_node)
    while next_node is not None:
        i += 1
        path[i] = np.array(next_node)
        curr_node = next_node
        next_node = preds.get(curr_node)
    return path

#******************************************************************************
def segments(tree):
    """ A generator of segments of a tree. The segments (non-standard) of a tree
        are a sequence of paths such that
            (1) the length of this sequence is minimized,
            (2) degree 1 and 2 nodes appear in exactly one path,
            (3) every node in the tree appears in at least one path.
        
        WARNING: this function is destructive. If you want to preserve 'tree' 
        you must copy it before calling the function.

        Args:
            tree: a networkx graph which is a tree. Will remove all nodes from 
            tree.

        Yields:
            Each segment. If tree is connected the longest segment will be first.
    """
    while tree:
        segment = longest_path(tree)
        tree.remove_nodes_from([tuple(n) for n in segment if tree.degree[tuple(n)] <= 2])
        yield segment
        
#******************************************************************************
def split_segment(segment, is_branch_point):
    """ Splits a segment based on its internal branch points. Each resulting 
        subsegment will contain at most two branch points, which are garunteed 
        to be at the ends of the segment. The length of each subsegment will be 
        at least 2 and will contain at least one non-branch point.
        
        If two subsegments share a branch point then this point will be 
        included in both subsegments. If there are many branch points in a row 
        then intermediate ones will not appear in any subsegment.
        
        Args:
            segment: a list
            is_branch_point: a callable taking one element of segment and 
            returning a Boolean
            
        Yields:
            The subsegments in order
    """
    lo = 0
    for hi, point in enumerate(segment):

        if is_branch_point(point):
            
            if hi - lo > 1 or (hi - lo == 1 and not is_branch_point(segment[lo])):
                #yield a copy since the segments overlap
                yield np.copy(segment[lo:hi+1])
            lo = hi 

    if hi - lo > 1 or (hi - lo == 1 and not (is_branch_point(segment[lo]) and is_branch_point(segment[hi]))):
        #yield a copy since the segments overlap
        yield np.copy(segment[lo:hi + 1])

#******************************************************************************
if __name__ == "__main__":    
    import matplotlib.pyplot as plt

    """print("Demonstrating segment generator function:")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    graph = nx.random_tree(10)
    nx.draw_networkx(graph, ax=ax)

    print('Segments generated from graph:')
    for segment in segments(graph):
        print(segment)

    plt.show()"""
    
    print("\nDemonstrating segment splitter:")
    s = [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
    test = lambda x : x == 1

    print(f"Segment = {s}. Branches occur at the 1's.")


    print(f"Split segment:")
    print(list(split_segment(s, test)))
