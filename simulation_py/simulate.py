import random
from collections import namedtuple
import networkx as nx

from scipy import stats
import numpy as np

# GraphVectors represent a position and location on a graph with edge weights.
# GraphVector(tail, head, edge_loc) represents the position on the edge 
# (tail, head) located edge_loc * weight(tail, head) away from tail
# (where 0 <= edge_loc 1)
GraphVector = namedtuple("GraphVector", ["tail", "head", "edge_loc"])

def initialise(graph, num_to_init):
    """ Initilises the positions of cells on the tissue graph. Chooses positions
        at random.

        Args:
            graph: a networkx graph (with edge weights) representing tissue
            num_to_init: the number of cells to initialise

        Returns:
            A list of GraphVectors of positions of the cells.
    """
    #choose edges for the inital positions of the cells
    edges = random.choices(list(graph.edges), k=num_to_init)
    
    initial = []
    for u, v in edges:

        #choose the direction the cell is moving at random
        if random.randint(0, 1) == 0: v, u = u, v
        
        #give the cell a random position along the chosen edge
        initial.append(GraphVector(u, v, random.uniform(0, 1)))

    return initial

def move_brownian(graph, gv, displacement):
    """ Generates a move for a cell moving with Brownian motion.

        Args:
            graph: a networkx graph (with edge weights) in which the cell is
            moving
            gv: a GraphVector representing the cells current position
            displacement: the amount by which to displace the cell (in the 
            graph's length units)

        Returns:
            A 2-tuple (new_gv, path), where new_gv is a GraphVector representing
            the cells new positition after the move, and path is a list of nodes
            which the cell moved through to arrive at its current position
    """
    #considers moves both forward of and reversed from its current direction
    positions = (
        get_moves(graph, gv.tail, gv.head, gv.edge_loc, displacement) +
        get_moves(graph, gv.head, gv.tail, 1 - gv.edge_loc, displacement)
    )

    return random.choice(positions)

def move_simple_forward(graph, gv, displacement, turning_probability):
    """ Generates a move for a cell that continues moving forward with probability
        1 - turning_probability.

        Args:
            graph: a networkx graph with edge weights in which the cell is moving
            gv: a GraphVector representing the cells current position
            displacement: the amount by which to displace the cell
            turning_probability: the probability the cell will reverse its direction

        Returns:
            A 2-tuple (new_gv, path), where new_gv is a GraphVector representing
            the cells new positition after the move, and path is a list of nodes
            which the cell moved through to arrive at its current position.
    """
    #decide whether to turn or not
    if random.uniform(0, 1) < turning_probability:
        return random.choice(get_moves(graph, gv.head, gv.tail, 1 - gv.edge_loc, displacement))
    else:
        return random.choice(get_moves(graph, gv.tail, gv.head, gv.edge_loc, displacement))

def move_normal_mixture(graph, gv, t_p, m_p, m1, s1, m2, s2):
    """ Generates a move for a cell where the displacement is drawn from one of 
        two normal distributions.

        Args:
            graph: a networkx graph with edge weights in which the cell is moving
            gv: a GraphVector representing the cells current position
            t_p: the probability the cell will reverse its direction
            m_p: the probability of choosing the first distribution
            m1: the mean of the first normal distribution
            s1: the standard deviation of the first normal distribution
            m2: the mean of the second normal distribution
            s2: the standard deviation of the second normal distribution

        Returns:
            A 2-tuple (new_gv, path), where new_gv is a GraphVector representing
            the cells new positition after the move, and path is a list of nodes
            which the cell moved through to arrive at its current position.
    """
    #generate a displacement from one of two normal distributions, where the 
    #distribution is chosen with probability m_p
    if np.random.uniform(0, 1) < m_p:
        displacement = abs(stats.norm.rvs(loc=m1, scale=s1))

    else:
        displacement = abs(stats.norm.rvs(loc=m2, scale=s2))
    
    return move_simple_forward(graph, gv, displacement, t_p)

def move_normal(graph, gv, t_p, m, s):
    return move_simple_forward(graph, gv, abs(stats.norm.rvs(loc=m, scale=s)), t_p)

def get_moves(graph, tail, head, edge_locn, displacement):
    """ For a cell located at GraphVector(tail, head, edge_locn) on graph, 
        returns a list of positions which are the given displacement away in the
        direction indicated by the GraphVector.

        Args:
            graph: a networkx graph with edge weights
            tail: a node in graph, which the cell is facing away from
            head: a node in graph which the cell is facing towards
            edge_locn: a float in [0, 1] which represents the proportion along 
            the edge (tail, head) the cell lies
            displacement: a displacement in the same units as the graph edge 
            weights
        
        Returns:
            A list of 2-tuples of the form (gv, path), where gv is a vector on 
            the graph and path is a list of nodes traversed on the path from 
            the current positon ot gv.
    """

    edge_length = float(graph.edges[(tail, head)]['weight'])
    dist_from_head = (1 - edge_locn) * edge_length

    #displacement is only enough to move some distance along the current edge
    if displacement < dist_from_head:
        return [(GraphVector(tail, head, edge_locn + displacement / edge_length), [])]

    #otherwise we will arrive at the node head
    locns = []
    for u in graph.adj[head]:
        
        #without reversing, continue moving along the edges incident to head with
        #the remaining displacement
        if u != tail:
            for gv, path in get_moves(graph, head, u, 0, displacement - dist_from_head):
                locns.append((gv, [head] + path))
    
    return locns

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    g = nx.Graph()

    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1.2)
    g.add_edge(0, 3, weight=2)
    g.add_edge(1, 4, weight=1)
    g.add_edge(0, 5, weight=1.1)



    gv1 = GraphVector(1, 0, 0.5)
    gv2 = GraphVector(4, 1, 0.6)


    print("gv1")
    print(get_moves(g, gv1.tail, gv1.head, gv1.edge_loc, 1))

    print("gv2")
    print(get_moves(g, gv2.tail, gv2.head, gv2.edge_loc, 1))
    print(get_moves(g, gv2.tail, gv2.head, gv2.edge_loc, 2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    nx.draw_networkx(g, ax=ax)
    plt.show()
