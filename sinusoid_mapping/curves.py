import itertools
from math import sqrt
from collections import namedtuple

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from scipy import optimize
from scipy import integrate
from scipy.interpolate import PPoly

import networkx as nx
from sinusoid_mapping import graph_tools

EPS = 1e-13

#******************************************************************************
def polyConst(c, domain=None, window=None):
    """Returns a constant polynomial taking value `c'"""
    p = Polynomial([c])
    return p.convert(domain=domain, window=window)

def intersect_domain(ps):
    return max(p.domain[0] for p in ps), min(p.domain[-1] for p in ps)

#******************************************************************************
class PolyCurve:
    """ This class represents parametized curves in any dimensional space where
        each component of the curve is a polynomial function
    """

    def __init__(self, fs):
        """
            Args:
                fs: a list of polynomial functions, one for each coordinate
        """
        self.fs = list(fs)

        #set domain of all coordinates to the largest domain on which all are defined
        self.domain_lo, self.domain_hi = intersect_domain(self.fs)

    def __str__(self):
        s = ", ".join(str(f) for f in self.fs)
        return f"({s})"

    def __repr__(self):
        s = ", ".join(repr(f) for f in self.fs)
        return f"{type(self).__name__}([{s}])"

    def __call__(self, t):
        """ Returns the value of the curve at t"""
        return np.array([f(t) for f in self.fs])
    
    def __iter__(self):
        return iter(self.fs)

    def max_degree(self):
        """ Returns the degree of the polynomial component with highest degree
        """
        return max(len(f.coef) for f in self.fs) - 1

    def dim(self):
        """ Gets the dimension in which this curve lives."""
        return len(self.fs)

    def points(self, num_points=50):
        """ Gets a specified number of points along the curve, equally spaced 
            (according the the parametization)
                
            Args:
                num_points: the number of points to generate
            Returns:
                An array of points along this curve.
        """

        points = np.empty((num_points, self.dim()), dtype=float)

        for i, t in enumerate(np.linspace(self.domain_lo, self.domain_hi, num=num_points)):
            points[i] = self(t)
        return points

    def deriv(self):
        """Calculates the derivative with respect to the parametization variable
        of the curve.

        Returns:
            A PolyCurve object representing the derivative of this curve
        """
        return PolyCurve([f.deriv() for f in self.fs])

    def length(self, lo=None, hi=None):
        """ Calculates the length along some section of this curve. If the
            optional arguments are not specified then the length of the whole 
            curve is returned.

            Args:
                lo: the value of the parameter at the start of the section of 
                curve 
                hi: the value of the parameter at the end of the section of curve
            Returns:
                A 2-tuple (y, err) where y is the length of the curve between lo 
                and hi (or the length of the whole curve if these are not 
                specifed) and err is an estimate of the absolute error in the 
                result.
        """
        if lo is None:
            lo = self.domain_lo
        if hi is None:
            hi = self.domain_hi
        
        #intergrand is the magnitude of the derivative (the speed)
        integrand = lambda t: sqrt(sum(df(t) ** 2 for df in self.deriv()))
        val, err = integrate.quad(integrand, lo, hi)
        return abs(val), err

    def path_integral(self, g, lo=None, hi=None):
        """Computes the path integral of g along a section of this curve. If
            lo (hi) is not supplied then the integral uses the start (end) of 
            curve.
        
            Args:
                g: a scalar valued function which takes as many inputs as the 
                dimension of this curve
                lo: the value of the lower terminal of the integral
                hi: the value of the upper terminal of the integral 
            Returns:
                The value of the path integral along this curve from lo to hi        
        """
        if lo is None:
            lo = self.domain_lo
        if hi is None:
            hi = self.domain_hi

        integrand = lambda t: g(*self(t)) * sqrt(sum(df(t) ** 2 for df in self.deriv()))
        return integrate.quad(integrand, lo, hi)

    def closest_point_parameter(self, point, return_distance_sq=False):
        """ Given a point p, finds the values t such that the distance between
            self(t) and p is minimized.

            Args:
                point: a point in the same dimension as this curve
                return_distance_sq: an optional flag. If set to true the minimum
                distance-squared is returned as well
            
            Returns:
                An array of parameters t such that the Euclidean distance of
                self(t) to point is minimized.

                (optional) The distance-squared between self(t) and point
        """

        #curve and point must both live in the same dimension
        assert len(self.fs) == len(point)

        #the idea is to minimise the distance squared function, which is a 
        #polynomial 
        dist_sq = sum((f - p) ** 2 for f, p in zip(self.fs, point))

        #the minimum distance occurs when the derivative is zero
        d_dist_sq = dist_sq.deriv()
        roots = d_dist_sq.roots()
        
        #remove imaginary roots 
        roots = roots.real[abs(roots.imag) < EPS]

        #remove roots which lie outside of the domain of the curve
        roots = roots[roots >= self.domain_lo]
        roots = roots[roots <= self.domain_hi]
        
        #the closest points may occur at the extema or the end points of the domain
        closest_candidates = np.concatenate((roots, np.array([self.domain_lo, self.domain_hi])))

        #compute the distance to the point corresponding to each parameter
        dists = sum((f(closest_candidates) - p) ** 2 for f, p in zip(self.fs, point))

        #return all parameters for which the minimum distance occurs
        if return_distance_sq:
            min_d = dists.min() 
            return (closest_candidates[np.where(dists == min_d)], min_d)

        return closest_candidates[np.where(dists == dists.min())]
        
    def closest_points(self, point):
        #TODO: Remove this method
        raise PendingDeprecationWarning()

        """ Calculates the points on the curve which are closest to a specifed
            point 
            
            Args:
                point: the point from which to find the closest point(s)
            Returns:
                A list of points on the curve which are closest to 'point', 
                noting that more than one point can be the closest point.
        """
        #curve and point must both live in the same dimension
        assert len(self.fs) == len(point)
        
        #the idea is to minimise the distance squared function, which is a 
        #polynomial 
        dist_sq = sum((f - p) ** 2 for f, p in zip(self.fs, point))

        #the minimum distance occurs when the derivative is zero
        d_dist_sq = dist_sq.deriv()
        roots = d_dist_sq.roots()
        
        #remove imaginary roots 
        closest = roots.real[abs(roots.imag) < EPS]

        #filter out the zeros which lie outside of the domain of the curve
        closest = [z for z in closest if z >= self.domain_lo and z <= self.domain_hi]

        #if filtering has removed all closest points then the closest point lies
        #at the edges of the domain
        if len(closest) == 0:         
            low_dist = sum((x - p) ** 2 for x, p in zip(self(self.domain_lo), point))
            hi_dist = sum((x - p) ** 2 for x, p in zip(self(self.domain_hi), point))
            
            if low_dist == hi_dist:
                closest = [self.domain_hi, self.domain_lo]
            elif low_dist > hi_dist:
                closest = [self.domain_hi]
            else:
                closest = [self.domain_lo]
        return [self(z) for z in closest]

    def distance_from(self, point):
        """ Returns the distance between a specified point and this curve. Note
            that this method calls 'closest_points' internally.
        
            Args:
                point: the point to find the distance to 
            Returns:
                The minimum distance from point to this curve
        """
        return sqrt(self.distance_sq_from(point))

    def distance_sq_from(self, point):
        cpoint = self.closest_points(point)[0]
        return sum((p - q) ** 2 for p, q in zip(cpoint, point))

    @staticmethod
    def from_coefficients(coeffs, domains, windows):
        return PolyCurve((Polynomial(c, domain=d, window=w) for c, d, w in zip(coeffs, domains, windows)))

    @staticmethod
    def graph(fs, param_axis=0):
        """ Produces a curve which is the graph of a list of polynomials about
            the chosen parametization axis. That is, the identity polynomial
            is added to the list of polynomials as the selected axis and the
            resulting curve is returned.

            Args:
                fs: a non-empty list of polynomials
                param_axis: the axis about which to graph the polynomials
            Returns:
                The graph as a PolyCurve object.
        """
        domain = intersect_domain(fs)
        fs = fs[:param_axis] + [Polynomial.identity(domain=domain)] + fs[param_axis:] 
        return PolyCurve(fs)

    @staticmethod
    def least_squares_fit(points, param_index, degrees=5):
        """ Produces a curve to fit a collection of points by finding a 
            polynomial of best fit (using least squares regression) for each
            axis plotted against the selected parametization axis.
            
            Args:
                points: a collection of points
                param_index: the index of the axis used to parametize the curve
                degrees: an iterable used to select the degree of each fitted
                polynomial
                
            Returns:
                The curve fitting the points as a PolyCurve object.
        """

        dim = len(points[0])
        
        if type(degrees) == int:
            degrees = itertools.repeat(degrees)

        #ensure degrees is an iterator
        degrees = iter(degrees)

        #parametization axis index must be valid 
        assert param_index < dim
        assert dim > 1

        #seperate the selected parametization axis data from the rest of the data
        param_axis = points[:,param_index]
        axes_pre = points[:,:param_index].transpose()
        axes_post = points[:,param_index+1:].transpose()

        #fit a polynomial function in each of the remaining axis in terms of the 
        #chosen axis variable
        polys_pre = [Polynomial.fit(param_axis, a, next(degrees)) for a in axes_pre]
        polys_post = [Polynomial.fit(param_axis, a, next(degrees)) for a in axes_post]
        
        #need to make sure the domain and windows of the polynomials match up
        domain_lo = min(f.domain[0] for f in polys_pre + polys_post)
        domain_hi = max(f.domain[-1] for f in polys_pre + polys_post)
        domain = [domain_lo, domain_hi]

        window_lo = min(f.window[0] for f in polys_pre + polys_post)
        window_hi = max(f.window[-1] for f in polys_pre + polys_post)
        window = [window_lo, window_hi]

        for i in range(len(polys_pre)):
            polys_pre[i] = polys_pre[i].convert(domain=domain, window=window)
        for i in range(len(polys_post)):
            polys_post[i] = polys_post[i].convert(domain=domain, window=window)

        return PolyCurve(polys_pre + [Polynomial.identity(domain=domain, window=window)] + polys_post) 

#******************************************************************************
class PiecewisePolyCurve:
    """ This class represents parametized curves in any dimensional space where
        each component of the curve is a piecewise polynomial function
    """
    def __init__(self, polys, bpoints):
        """ Creates a SplineCurve.

            Args:
                polys: A list of numpy.polynomial.polynomial.Polynomial points.
                The ith polynomial represents the curve from bpoints[i] to 
                bpoints[i+1]. It is assumed that the domain of this polynomial
                is equal to [bpoints[i], bpoints[i+1]].
                bpoints: the 'break points' determining the points on the curve
                at which one polynomial curve transitions to another. 
                Must have length len(polys) + 1 and must be sorted.
        """
        self.curves = [PolyCurve(ps) for ps in polys]
        self.bpoints = bpoints
    
    def __call__(self, x):
        """ Returns the value of the curve at x"""
        i = self.curve_index(x)
        return self.curves[i](x)

    def __str__(self):
        out = ""
        for f, lo, hi in zip(self.curves, self.bpoints[:-1], self.bpoints[1:]):
            out += f"{[lo, hi]}:\t{str(f)}\n"
        return out

    def __repr__(self):
        out = "< PiecewisePolyCurve"
        for f, lo, hi in zip(self.curves, self.bpoints[:-1], self.bpoints[1:]):
            out += f"\t{[lo, hi]}:\t{repr(f)}\n"
        return out + "/>"

    def dim(self):
        """ Returns the dimension in which this curve is embedded"""
        return self.curves[0].dim()

    def domain(self):
        """ Returns the domain of the parameter space of this curve"""
        return [self.bpoints[0], self.bpoints[-1]]

    def curve_index(self, val):
        """ Gets the index of the PolyCurve defining this curve at val.

            Args:
                val: a parameter value

            Returns:
                The index of the PolyCurve such that val is in the domain of the
                PolyCurve
        """
        if val < self.bpoints[0] or val > self.bpoints[-1]:
            raise ValueError(f"Argument {val} lies outside domain {[self.bpoints[0], self.bpoints[-1]]}")
        
        for i, hi in enumerate(self.bpoints[1:]):
            if val <= hi:
                return i

    def reparameterise(self, domain_map):
        pass

    def connect_to_point(self, point, position, tspace_length=1):
        """ Connects this curve to a point by interpolating linearly.
        
            Args:
                point: the point to connect to (must be in same dimension as 
                the curve)
                position: either 0 or -1 to attach point to start or end of curve
                tspace_length: the length of the parameter space of the new 
                curve section.         
        """
        assert position == 0 or position == -1
        assert len(point) == self.dim()

        if position == 0:
            start = point
            end = self(self.bpoints[position])

            tstart = self.bpoints[position] - tspace_length
            tend = self.bpoints[position]

            self.bpoints = np.insert(self.bpoints, position, tstart)
        else:
            start = self(self.bpoints[position])
            end = point

            tstart = self.bpoints[position]
            tend = self.bpoints[position] + tspace_length
            
            self.bpoints = np.append(self.bpoints, tend)

        coefs = [[s - tstart * (e - s) / (tend - tstart), (e - s) / (tend - tstart)]
                 for s, e in zip(start, end)]

        ps = [Polynomial(c, domain=[tstart, tend], window=[tstart, tend]) 
              for c in coefs]
        
        self.curves.insert(position, PolyCurve(ps))

    def length(self, lo=None, hi=None):
        """ Computes the length of (a section of) this curve. 

            Args:
                lo: the parameter of the point at which to start measuring 
                the length. If not set then lo = self.bpoints[0]. Assumed that
                self.bpoints[-1] >= hi > lo >= self.bpoints[0]
                hi: the parameter of the point at which to stop measuring the 
                length. If not set then hi = self.bpoints[-1]. Assumed that
                self.bpoints[-1] >= hi > lo >= self.bpoints[0]
            
            Returns:
                A 2-tuple (y, err) where y is the length of the curve between 
                self(lo) and self(hi) and err is an estimate of the absolute 
                error in y.
        """
        if lo is None:
            lo = self.bpoints[0]
        if hi is None:
            hi = self.bpoints[-1]

        lo_i = self.curve_index(lo)
        hi_i = self.curve_index(hi)

        #section lies entirely within one part of this curve
        if lo_i == hi_i:
            y, err = self.curves[lo_i].length(lo=lo, hi=hi)
            return y, np.longdouble(err)

        y, err = self.curves[lo_i].length(lo=lo) 
        err = np.longdouble(err)

        for i in range(lo_i + 1, hi_i):
            curr_y, curr_err = self.curves[i].length()
            y += curr_y
            err += np.longdouble(curr_err)
        
        curr_y, curr_err = self.curves[hi_i].length(hi=hi)
        y += curr_y
        err += np.longdouble(curr_err)
        return y, err

    def path_integral(self, g, lo=None, hi=None):
        """ Computes the path integral of a scalar function g along (a section 
            of) this curve.

            Args:
                g: a callable taking points in the dimension of this curve 
                and returning something float-like.
                lo: The lower terminal of the integral. If not set then 
                lo = self.bpoints[0]. Assumed that 
                self.bpoints[-1] >= hi > lo >= self.bpoints[0]
                hi: The upper terminal of the integral. If not set then 
                hi = self.bpoints[-1]. Assumed that
                self.bpoints[-1] >= hi > lo >= self.bpoints[0]
        
            Returns:
                The value of the path integral from lo to hi.
        """

        if lo is None:
            lo = self.bpoints[0]
        if hi is None:
            hi = self.bpoints[-1]
        
        lo_i = self.curve_index(lo)
        hi_i = self.curve_index(hi)
 
        return (self.curves[lo_i].path_integral(g, lo=lo) 
                + sum(self.curves[i].path_integral(g) for i in range(lo_i + 1, hi_i)) 
                + self.curves[hi_i].path_integral(g, hi=hi))

    def closest_point_parameter(self, point, all_points=True, return_distance_sq=False):
        """ Computes the values t such that the distance from self(t) to point
            is minimum.

            Args:
                point: a point in the same dimension as this curve.
                all_points: (optional) set to False to only return a single value
            
            Returns:
                An array of values which satisfy distance(self(t), point) being
                minimum, or if all_points is set to false, just one value 
        """
        fs_iter = iter(self.curves)
        f = next(fs_iter)
        min_ts, min_dist = f.closest_point_parameter(point, return_distance_sq=True)
        for f in fs_iter:

            ts, dist = f.closest_point_parameter(point, return_distance_sq=True)
            if abs(min_dist - dist) < EPS:
                min_ts = np.concatenate((min_ts, ts))
            elif min_dist > dist:
                min_dist = dist
                min_ts = ts
        
        #if a closest point occurs at end of an interval then two polynomials
        #will report the same closest point parameter, so return unique values
        if all_points: 
            if return_distance_sq:
                return min_dist, np.unique(min_ts)
            else:
                return np.unique(min_ts)
        if return_distance_sq:
            return min_dist, min_ts[0]
        else:
            return min_ts[0]

    @staticmethod
    def create_from_spline_tck(tck):
        """ Creates a PiecewisePolyCurve object from a tuple tck representing
            a spline (such as as returned by scipy.interpolate.splprep).

            Args:
                tck: A tuple (t, cs, k) where t is a list of knots of the spline
                (parameters where one polynomial changes to another), cs is a 
                list of B-spline basis coefficients with length equal to the 
                dimension of the space containing this spline, and k is the
                degree of the spline (an integer)
            
            Returns:
                A PiecewisePolyCurve object representing the spline.
        """

        t, cs, k = tck
        
        ppolys = [trim_ppoly(PPoly.from_spline((t, c, k))) for c in cs]

        #PPoly.from_spline returns piecewise polynomials with identical 
        #break points for the same knots t.
        bpoints = ppolys[0].x

        ps = []
        for bi in range(len(bpoints) - 1):
            ps.append([ppoly_to_polynomial(ppoly, bi) for ppoly in ppolys])

        """
        ps = len(ppolys) * [[]]
        for ax, ppoly in enumerate(ppolys):


            for bi in range(len(bpoints) - 1):

                ps[ax].append(ppoly_to_polynomial(ppoly, bi))
        """
        return PiecewisePolyCurve(ps, bpoints)

#******************************************************************************
class GraphCurve:

    def __init__(self):
        """ Creates an empty GraphCurve."""
        self.curve_graph = nx.DiGraph()

    def __call__(self, id_param):
        """ Evaluates the curve at id_param.

            Args:
                id_param: a tuple (curve_id, param) where curve_id specifies 
                the curve and param the value at which the evaluate that curve.

            Returns:
                The value of the curve with id curve_id at param.
        """
        curve_id, param = id_param
        return self.curve_graph.nodes[curve_id]['curve'](param)

    def parameter_space(self):
        """ Returns an iterator representing the parameter space of this curve.

            Yields:
                Tuples of the form (curve_id, [lo, hi]) where curve_id specifies
                a curve and [lo, hi] the domain of this curve. A call
                self((cid, t)) is valid if and only if cid == curve_id and
                lo <= t <= hi for some tuple (curve_id, [lo, hi]) produced by 
                this generator
        """
        return ((cid, self.curve_graph.nodes[cid]['curve'].domain()) for cid in self.curve_graph.nodes)

    def branches(self):
        """ Returns an iterator on the branches of this GraphCurve"""
        return (self.curve_graph.nodes[n]['curve'] for n in self.curve_graph.nodes)

    def num_branches(self):
        """ Returns the number of branches of this GraphCurve"""
        return len(self.curve_graph.nodes)

    def length(self, id_param_start, id_param_end):
        """ Calculates the length between two positions along the GraphCurve
        
            Args:
                id_param_start: specifies the starting position 
                id_param_end: specifies the ending position
            
            Returns:
                A 2-tuple (y, err) where y is the length between id_param_start 
                and id_param_end and err is an estimate of the absolute error in
                y.
        """
        start_id, start_param = id_param_start
        end_id, end_param = id_param_end

        curve_seq = nx.bidirectional_shortest_path(self.curve_graph, start_id, end_id)
        
        length = 0
        err = np.longdouble(0)

        curve_start = start_param
        for curr_curve_id, next_curve_id in zip(curve_seq, curve_seq[1:]):
            curve_end = self.curve_graph.edges[(curr_curve_id, next_curve_id)]['t']

            curr_length, curr_err =  self.curve_graph.nodes[curr_curve_id]['curve'].length(curve_start, curve_end)

            length += curr_length
            err += np.longdouble(curr_err)

            curve_start = self.curve_graph.edges[(next_curve_id, curr_curve_id)]['t']

        curr_length, curr_err = self.curve_graph.nodes[end_id]['curve'].length(curve_start, end_param)

        length += curr_length
        err += np.longdouble(curr_err)

        return length, err 

    def directions(self, id_param_start, id_param_end):
        """ Finds the direction which to leave in, and to arrive from, on the 
            path between id_param_start and id_param_end. Direction is defined
            at each point by the local parametization, so directions are only 
            meaningful within the same curve. 

            Args:
                id_param_start: specifies the starting position (valid argument
                for __call__)
                id_param_end: specifies the ending position (valid argument
                for __call__)
            
            Returns:
                A 2-tuple (sdir, edir) where sdir sign of the change in the 
                parameter at the start of the path, and similarly edir is the
                sign of the change in the parameter at the end of the parth.
        """
        start_id, start_param = id_param_start
        end_id, end_param = id_param_end

        curve_seq = nx.bidirectional_shortest_path(self.curve_graph, start_id, end_id)

        if len(curve_seq) == 1:
            start_dir = np.sign(end_param - start_param)
            end_dir = start_dir
            
        else:

            #the parameter at which the first curve joints the 2nd curve
            scnd_param = self.curve_graph.edges[(curve_seq[0], curve_seq[1])]['t']

            #the parameter at which the 2nd last curve joins the last curve
            scnd_last_param = self.curve_graph.edges[(curve_seq[-1], curve_seq[-2])]['t']

            start_dir = np.sign(scnd_param - start_param)
            end_dir = np.sign(end_param - scnd_last_param)

        return start_dir, end_dir

    def closest_point_parameter(self, point):
        """ Returns the parameter (a suitible argument for self.__call__) which
            specifies a point on this curve which is closest to point.

            Args:
                point: a point in the same dimension as this curve
            
            A value x such that the distance from point to self(x) is minimal
        """
        dists = {
            cid: self.curve_graph.nodes[cid]['curve'].closest_point_parameter(
                point, 
                all_points=False, 
                return_distance_sq=True
                ) 
            for cid in self.curve_graph.nodes}

        min_id = min(dists, key=dists.get)
        return min_id, dists[min_id][1]

    def closest_point(self, point):
        return self(self.closest_point_parameter(point))

    def is_end_point(self, param):
        curve_id, p = param

        #this parameter may be the end point of this curve, but not the
        #whole GraphCurve
        for adj_curve_id in self.curve_graph.adj[curve_id]:
            
            #if p is equal to the parameter of a branch point then it is not and end point
            if abs(p - self.curve_graph.edges[(curve_id, adj_curve_id)]['t']) < EPS:
                return False

        #otherwise it is an end point iff it is at the end of its curve's domain
        p_lo, p_hi = self.curve_graph.nodes[curve_id]['curve'].domain()
        return (abs(p - p_lo) < EPS) or (abs(p - p_hi) < EPS)

    @staticmethod
    def create_from_nng(nng, interpolator):
        """ Creates a GraphCurve object from a nearest neighbour graph of points

            The nearest neighbour graph is assumed to be a tree.

            Args:
                nng: a networkx Graph object with nodes as points and edges 
                interpolator: a callable which accepts a numpy array of shape 
                (N, d) representing N points in dimension d and returns output
                suitible for Polycurve.create_from_spline_tck input.
                
            Returns:
                A GraphCurve object fitting the points in nng
        """
        gc = GraphCurve()

        #the points at which nng (a tree) branches will become edges in the 
        #GraphCurve
        branch_points = {n for n in nng if nng.degree[n] > 2}

        #we fit the longest branches first, and join shorter branches onto them
        #we term the branch being joined to a 'trunk'.
        trunks = dict()

        #nodes are integers, the curve is an attribute of the node
        seg_ids = itertools.count()

        for s in graph_tools.segments(nng):

            #if the branch cuts a trunk we need to split it it into more branches
            splitter = lambda p : tuple(p) in trunks

            for branch in graph_tools.split_segment(s, splitter):

                branch_id = next(seg_ids)
                edges = []

                #find branch points on this branch
                curr_bpts = branch_points.intersection(tuple(b) for b in branch)

                for trunk_join in curr_bpts.intersection(trunks):
                    #find the end of the branch we are joining to the trunk.
                    #graph_tools.split_segments garuntees this will be at one
                    #end
                    end = 0 if fseq_equal(branch[0], trunk_join) else -1

                    trunk_id = trunks[trunk_join]
                    trunk = gc.curve_graph.nodes[trunk_id]['curve']

                    #join it to a point on the trunk which is closest to the 
                    #joining end of the branch
                    trunk_t = trunk.closest_point_parameter(branch[end], all_points=False)

                    #replace end point with joining point
                    branch[end] = trunk(trunk_t)

                    edges.append((trunk_id, trunk_t, end))

                trunks.update({p: branch_id for p in curr_bpts.difference(trunks)})

                curve = PiecewisePolyCurve.create_from_spline_tck(interpolator(branch))
                gc.curve_graph.add_node(branch_id, curve=curve)

                for other_id, in_param, end in edges:
                    gc.curve_graph.add_edge(other_id, branch_id, t=in_param)
                    gc.curve_graph.add_edge(branch_id, other_id, t=curve.domain()[end])
        return gc

#*******************************************************************************
# Helper functions for dealing with intermediate scipy.interpolate.PPoly objects

def trim_ppoly(ppoly):
    """ Removes repeated length-zero intervals at ends of a ppoly object, as 
        often produced by B-spline interpolation.

        Args:
            ppoly: A scipy.interpolate.PPoly object.
        
        Returns:
            A new PPoly object which is identical to ppoly, except with the 
            repeated length-zero intervals at either end removed
    """
    i = 0
    while ppoly.x[i] == 0:
        i += 1

    j = len(ppoly.x) - 1
    while ppoly.x[j] == 1:
        j -= 1
    return PPoly(ppoly.c[:,i-1:j+1], ppoly.x[i-1:j+2])

def ppoly_to_polynomial(ppoly, bi):
    """ Converts the bi^th polynomial in a PPoly object to a Polynomial object.

        Args: 
            ppoly: the PPoly object to convert from
            bi: the breakpoint index

        Returns:
            A Polynomial object representing the polynomial from ppoly.x[bi] to
            ppoly.x[bi + 1]. It has domain and window [ppoly.x[bi], ppoly.x[bi+1]]
    """
    domain = ppoly.x[bi:bi + 2]

    #The polynomial in [poly_x.x[bi], poly_x.x[bi + 1]] is stored in the in the 
    #basis (x - poly_x.x[bi]) ** n, so lcoefs ('local coefficients') are the 
    #coefficients of these polynomials. Higher degree coefficients are stored
    #at the start of the list, so we flip this list.
    lcoefs = np.flip(ppoly.c[:,bi])

    #this is the polynomial `x`
    xp = Polynomial.identity(domain=domain, window=domain)
    return sum(c * (xp - ppoly.x[bi]) ** n for n, c in enumerate(lcoefs))

#*******************************************************************************
#Other helper functions
def fseq_equal(ps, qs, eps=EPS):
    """ Checks if two sequences of floats are equal up to eps precision

        Args:
            ps, sq: two sequences of floats
            eps: (optional) floating point precision level

        Returns:
            True if ps and sq are equal up to eps precison and False otherwise
    """
    return all(abs(p - q) < eps for p, q in zip(ps, qs))

#*******************************************************************************
# Functions which test or demonstrate parts of this module
def test2d():
    import matplotlib.pyplot as plt

    plt.ion()

    f = lambda t : [t, 1 - t ** 2] if -2 <= t <= 0 else [t, 1 - t]
    f_domain = [-2, 1]
    f_bpts = [-2, 0, 1]
    f_polys = [
        [
            Polynomial([0, 1], domain=[-2, 0], window=[-2, 0]), 
            Polynomial([1, 0, -1], domain=[-2, 0], window=[-2, 0])
        ], [
            Polynomial([0, 1], domain=[0, 1], window=[0, 1]),
            Polynomial([1, -1], domain=[0, 1], window=[0, 1]),   
        ]
    ]
    f_curve = PiecewisePolyCurve(f_polys, f_bpts)

    g = lambda t : [t, t] if 0.5 <= t <= 1 else [t ** 2, t]
    g_domain = [0.5, 2]
    g_bpts = [0.5, 1, 2]
    g_polys = [
        [
            Polynomial([0, 1], domain=[0.5, 1], window=[0.5, 1]),
            Polynomial([0, 1], domain=[0.5, 1], window=[0.5, 1]),
        ], [
            Polynomial([0, 0, 1], domain=[1, 2], window=[1, 2]),
            Polynomial([0, 1], domain=[1, 2], window=[1, 2]),
        ]
    ]
    g_curve = PiecewisePolyCurve(g_polys, g_bpts)

    h = lambda t : [t, (t + 1) ** 3] if -2 <= t <= -1 else [-2, t]
    h_domain = [-2.5, -1]
    h_bpts = [-2.5, -2, -1]
    h_polys = [
        [
            Polynomial([-2], domain=[-2.5, -2], window=[-2.5, 2]),
            Polynomial([0, 1], domain=[-2.5, -2], window=[-2.5, 2])
        ], [
            Polynomial([0, 1], domain=[-2, -1], window=[-2, -1]),
            Polynomial([1, 3, 3, 1], domain=[-2, -1], window=[-2, -1])
        ]
    ]
    h_curve = PiecewisePolyCurve(h_polys, h_bpts)
    
    f_pts = np.array([f(t) for t in np.linspace(f_domain[0], f_domain[1], num=200)]).transpose()
    g_pts = np.array([g(t) for t in np.linspace(g_domain[0], g_domain[1], num=200)]).transpose()
    h_pts = np.array([h(t) for t in np.linspace(h_domain[0], h_domain[1], num=200)]).transpose()

    joins = [(0.5, 0.5), (-1, 0)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(f_pts[0], f_pts[1], c='b', label="0: f(t)")
    ax.plot(g_pts[0], g_pts[1], c='g', label="1: g(t)")
    ax.plot(h_pts[0], h_pts[1], c='r', label="2: h(t)")

    for pt in joins:
        ax.annotate(str(pt), pt)
        ax.scatter(pt[0], pt[1], c='k')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

    curve = GraphCurve()
    curve.curve_graph.add_node(0, curve=f_curve)
    curve.curve_graph.add_node(1, curve=g_curve)
    curve.curve_graph.add_node(2, curve=h_curve)
    
    curve.curve_graph.add_edge(0, 1, t=0.5)
    curve.curve_graph.add_edge(1, 0, t=0.5)

    curve.curve_graph.add_edge(0, 2, t=-1)
    curve.curve_graph.add_edge(2, 0, t=-1)

def test_plot_track_curves(track_file, show_track=True, show_nng=True):
    import matplotlib.pyplot as plt
    from scipy import interpolate

    import graphing
    import config
    from tracks import Track

    tracks = Track.from_spots_file(config.test_spots)

    AV_SINUSOID_DIAMETER = 6

    for track in tracks.values():


        sinusoid_points = graph_tools.simplify_points(track.points(), AV_SINUSOID_DIAMETER)
        nng = graph_tools.nearest_neighbor_graph(sinusoid_points)
        nng_copy = nng.copy()
        curve = GraphCurve.create_from_nng(nng, interpolate_branch)

        print(f"Number of branches: {curve.num_branches()}")

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if show_track: graphing.mpl_scatter3d(ax, track.points(), color='k')
        if show_nng: graphing.mpl_r3graph(ax, nng_copy)

        num_curve_points = 50
        for branch in curve.branches():
            
            start, stop = branch.domain()
            line = np.linspace(start, stop, num=num_curve_points)

    
            pts = [branch(x) for x in line]
            graphing.mpl_line3d(ax, pts, color='g')

        graphing.mpl_set_labels(ax)
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    from scipy import optimize
    from scipy import interpolate

    pts = np.random.uniform(0, 1, size=(10, 3))

    tck, u = interpolate.splprep(pts.T)

    p = PiecewisePolyCurve.create_from_spline_tck(tck)

    lo, hi = p.domain()
    x = np.random.uniform(lo, hi)


    length, err = p.length(lo, x)
    print(f"domain: {[lo, hi]}")
    print(f"True x = {x}")

    f = lambda t : length - p.length(lo, t)[0]

    ans = optimize.fsolve(f, (hi - lo) / 2)

    print(ans)
