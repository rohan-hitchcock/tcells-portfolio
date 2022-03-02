import bisect
import csv
from collections import namedtuple
from collections import defaultdict as dd

import numpy as np

TrackPoint = namedtuple("TrackPoint", ["time", "position"])

TIDS = 'track_ids'
EDGES = 'track_edges'
TIMES = 'times'
POSITIONS = 'positions'

#******************************************************************************

class Track:

    def __init__(self):
        """Initialises an empty track."""
        self.track = []

    def __len__(self):
        return len(self.track)
    
    def __str__(self):
        return "\n".join(str(t) + ", " + str(p) for t, p in self.track)

    def __bool__(self):
        return bool(self.track)

    def __iter__(self):
        return iter(self.track)

    def __getitem__(self, i):
        """Returns the ith track point"""
        return self.track[i]

    def get_slice(self, start, stop):
        """ Implements slicing so that the return value is a Track object. The 
            usual Python slicing rules apply.

            Args:
                start: an integer for the start of the slice, or None to go from 0
                stop: an integer for the end of the slice, or None to go to end

            Returns:
                The described slice.
        """
        if start is None: start = 0
        if stop is None: stop = len(self)

        sliced = Track()
        sliced.track = self.track[start:stop]
        
        return sliced

    def points(self):
        return np.array([p for t, p in self.track])

    def index(self, t):
        """ Gets the index of the TrackPoint having time equal to t. Returns 
            None if no such TrackPoint exists
            
            Args:
                t: a time point
            Returns:
                The index of the TrackPoint having time t, or None if no such 
                Track point exists.
        """
        l = 0
        r = len(self.track) - 1
        while l <= r:
            m = (l + r) // 2
            if self.track[m].time < t:
                l = m + 1
            
            elif self.track[m].time > t:
                r = m - 1

            else:
                return m      
        return None

    def duration(self):
        return self[-1].time - self[0].time 

    def at_time(self, t):
        """ Gets the position of the track at time t.

            Args:
                t: a time point
            Returns:
                The position of the track at time t, or None if the track does
                not exist at this time
        """
        i = self.index(t)

        if i is None:
            return None
        
        return self.track[i].position

    def insert(self, time, position):
        """ Adds a TrackPoint with the specified time and position to the Track.
            A track can only have one position at each time, so if there is 
            already a TrackPoint with the specified time this is overwritten.
            
            Args:
                time: the time value to add
                position: a position to add
        """
        i = self.index(time)
        if i is None:
            bisect.insort(self.track, TrackPoint(time, position))
        else:
            self.track[i] = TrackPoint(time, position)

    def pairwise(self, delta):
        """ Produces a generator which iterates through the track two TrackPoints
            at a time. The generator will produce every pair of points which
            differ in time by delta. Each pair given in increasing time order.
            
            Args:
                delta: the difference in time between points generated
            
            Yields:
                Pairs of points which differ in time by delta
        """
        lo = 0
        hi = 1
        while hi < len(self):
            
            #time between indices is correct, so output data
            if self.track[hi].time - self.track[lo].time == delta:
                yield self.track[lo], self.track[hi]
                lo += 1
                hi += 1
            
            #indecies are too close in time, so increase by increasing hi
            elif self.track[hi].time - self.track[lo].time < delta:
                hi += 1
            
            #indicies are too far apart in time, so decrease by increasing lo
            else:
                lo += 1
                if hi == lo:
                    hi += 1

    def subtrack_filter(self, predicate):
        """ Filters the points in a track according to the provided predicate.

            Args:
                predicate: A function which takes a TrackPoint and returns a 
                Boolean

            Returns:
                A 2-tuple (true_subtrack, false_subtrack), where true_subtrack
                is a Track contain the points for which the predicate function 
                is true, and false_subtrack contains the points for which the 
                predicate function is false.
        """

        true_subtrack = Track()
        false_subtrack = Track()
        
        for track_point in self:

            if predicate(track_point):

                true_subtrack.insert(*track_point)
            else:
                false_subtrack.insert(*track_point)

        return true_subtrack, false_subtrack

    @staticmethod
    def from_spots_file(spotsfile):
        """" Produces a dictionary of tracks keyed by track id from the data 
             easily obtained from an Imaris spots file.

             Args:
                spotsfile: The path of an .npz file containing the date of an
                imaris spots object.

            Returns:
                a (default) dictionary of tracks keyed by track ids
        """

        tracks = dd(Track)

        spots = np.load(spotsfile)
        
        for i, track_id in enumerate(spots[TIDS]):
            start_i, end_i = spots[EDGES][i]      
            tracks[track_id].insert(
                spots[TIMES][start_i],
                spots[POSITIONS][start_i]
            )
            tracks[track_id].insert(
                spots[TIMES][end_i], 
                spots[POSITIONS][end_i]
            )

        return tracks

    @staticmethod
    def from_csv(csv_file,  tid='id', time='t', x='x', y='y', z='z'):
        """ Produces a dictionary of tracks keyed by track id from track data 
            stored in a csv file. 

            Args:
                csv_file: The path to a .csv file containing track data
                tid: the name of the track id column in the file 
                time: the name of the column in the file
                x: the name of the column storing the x-coordinate
                y: the name of the column storing the y-coordinate
                z: the name of the column storing the z-coordinate

            Returns:
                A (default) dictionary of tracks keyed by track id
        """

        tracks = dd(Track)
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for line in reader:
                tracks[line[tid]].insert(
                    time=int(line[time]),
                    position=np.array([float(line[x]), float(line[y]), float(line[z])])
                )
        return tracks

    @staticmethod
    def from_iterable(l):
        """ Creates a Track object from a iterable of length 2 array-like values
            of the form (time, positon) where time is a float-like value 
            representing the time point, and position is some kind of object
            representing a position (i.e. a tuple, list, numpy array).

            Args:
                l: a iterable as described, with all unique time values

            Returns:
                A track object with all the objects in l
        """

        track = Track()
        track.track = [TrackPoint(t, p) for t, p in l].sort()
        return track

if __name__ == "__main__":
    track = Track()
    track.insert(0, [0, 1, 2])
    track.insert(1, [0, 0, 0])
    track.insert(3, [0, 1, 0])
    print(track)

    print(track[1])
    print(Tracktrack[1:])    