import numpy as np
from skimage import measure
from collections import namedtuple

BoundingBox = namedtuple("BoundingBox", ['xmin', 'xmax', 
                                         'ymin', 'ymax', 
                                         'zmin', 'zmax'])

VDATA = 'data'
BBOX_MAX_X = 'max_x'
BBOX_MIN_X = 'min_x'
BBOX_MAX_Y = 'max_y'
BBOX_MIN_Y = 'min_y'
BBOX_MAX_Z = 'max_z'
BBOX_MIN_Z = 'min_z'
NUM_VOX_X = 'size_x'
NUM_VOX_Y = 'size_y'
NUM_VOX_Z = 'size_z'

def is_inside_bbox(bbox, point):
    x, y, z = point 
    return ((bbox.xmin <= x) and (bbox.xmax >= x) and 
            (bbox.ymin <= y) and (bbox.ymax >= y) and 
            (bbox.zmin <= z) and (bbox.zmax >= z))

class Surface:

    def __init__(self, vdata, bbox, vdim):
        """ Constructor

            Args:
                vdata: A time-indexed collection of 3d numpy arrays. Each cell
                in the array is a voxel. If the voxel is <0 then this point is 
                outside the surface, =0 on surface and >0 inside the surface.
                bbox: A time indexed collection of minimal bounding boxes for 
                the surface orthogonal to xyz coordinates
                vdim: A time indexed collection of the x, y, z dimensions of a 
                voxel
        """
        self.vdata = vdata
        self.bbox = bbox
        self.vdim = vdim

    def get_bbox(self, time_index):
        """ Gets the bounding box for the surface at a specificed time.

            Args:
                time_index: The time index of the bounding box to get.

            Returns:
                A BoundingBox for the specified time index.
        """
        return BoundingBox(
                self.bbox.xmin[time_index], self.bbox.xmax[time_index], 
                self.bbox.ymin[time_index], self.bbox.ymax[time_index],
                self.bbox.zmin[time_index], self.bbox.zmax[time_index])
    
    def get_vdim(self, time_index):
        """ Gets the voxel dimensions for the surface at a specified time.

            Args:
                time_index: The time index of the dimensions to get

            Returns:
                A 3-tuple for the x, y, z dimensions of the voxel at that time.
        """
        x, y, z = self.vdim
        return x[time_index], y[time_index], z[time_index]

    def trisurf(self, time_index, located=True):
        """ Gets a triangulated surface representation for the surface at 
            the specifed time.
            
            Args:
                time_index: The time index to get the surface at.
                located: Set to false to get the trisurf near the origin
            
            Returns:
                A 5-tuple consisting of:
                    - An array of vertices (3d points) for the triangulation
                    - An array of groups of 3 indices, which index vertices, 
                    representing the trianglular faces 
                    - An array of the normal vectors at each vertex
                    - An array of values derived from the minimum local value of the data
        """

        #get bounding box and voxel dimensions at this time
        curr_bbox = self.get_bbox(time_index)
        x_vdim, y_vdim, z_vdim = self.get_vdim(time_index)

        verts, faces, normals, values = measure.marching_cubes_lewiner(self.vdata[time_index], level=0, spacing=(x_vdim, y_vdim, z_vdim))

        if located:
            #place the surface in the correct place in 3d space
            verts += [curr_bbox.xmin, curr_bbox.ymin, curr_bbox.zmin]

        return verts, faces, normals, values

    def is_inside(self, point, time_index):
        """ Tests whether a point is inside the surface at the specified time.
            
            Args:
                point: The 3d point to test.
                time_index: The time at which to test
            
            Returns:
                True if the specified point is inside the vertex, and false 
                otherwise.
        """

        #gets the current bounding box and voxel dimensions
        curr_bbox = self.get_bbox(time_index)
        x_vdim, y_vdim, z_vdim = self.get_vdim(time_index)

        #if the point is outside the bounding box then exit early
        if not is_inside_bbox(curr_bbox, point):
            return False

        x, y, z = point
        
        #convert coordinates to the index of the voxel it lies in
        x_i = int(round((x - curr_bbox.xmin) / x_vdim))
        y_i = int(round((y - curr_bbox.ymin) / y_vdim))
        z_i = int(round((z - curr_bbox.zmin) / z_vdim))

        #values above zero lie inside the surface
        return self.vdata[time_index][x_i][y_i][z_i] > 0

    def points(self, time_index, located=True):
        """ Gets some of points inside the surface at the specified time.
        
            Args:
                time_index: the time at which to get the points.
                
            Returns:
                An array of points which are inside the surface.
        """

        curr_bbox = self.get_bbox(time_index)
        curr_mins = np.array([curr_bbox.xmin, curr_bbox.ymin, curr_bbox.zmin])

        curr_vdim = np.array(self.get_vdim(time_index))

        curr_vdata = self.vdata[time_index]

        points = []
        for x_i in range(len(curr_vdata)):

            for y_i in range(len(curr_vdata[x_i])):

                for z_i in range(len(curr_vdata[x_i][y_i])):

                    #voxels less than / equal to zero are outside / on surface
                    if curr_vdata[x_i][y_i][z_i] <= 0:
                        continue
                          
                    p_i = np.array([x_i, y_i, z_i])
                    
                    #adds a point at the centre of the current voxel
                    if located:
                        points.append(p_i * curr_vdim + curr_mins + (curr_vdim / 2))
                    else:
                        points.append(p_i * curr_vdim + (curr_vdim / 2))

        return np.array(points)

    def vert_line_test(self, time_index, dom_i, codom_i):
        """ Performs a 'vertical line test' on the surface projected onto the 
            two selected axies. Specifically it checks whether every vertical 
            line crosses the projection of the surface at most once.
            
            Args:
                dom_i: the index of the domain axis for the projection
                codom_i: the index of the codomain axis for the projection
                
            Returns:
                Returns True if every vertical line of the projection of the
                surface onto the dom_i-codom_i plane crosses the surface at 
                most once, and false otherwise
        """
        assert dom_i != codom_i
        
        #gets a view of the voxel data so the domain and codomain are on the x 
        #ad y axes
        voxel_view = np.swapaxes(self.vdata[time_index], 0, dom_i)
        voxel_view = np.swapaxes(voxel_view, 1, dom_i)

        for x_slice in voxel_view:
            
            x_slice_iter = iter(x_slice)

            y_slice = next(x_slice_iter)
            prev_is_inside = any(y_slice > 0)

            boundary_count = 1 if prev_is_inside else 0
            for y_slice in x_slice_iter:
            
                curr_is_inside = any(y_slice > 0)

                #we count the number of transitions to fully outside to fully 
                #inside
                if curr_is_inside != prev_is_inside:
                    boundary_count += 1

                    #more than 3 boundary crossings means there are two 
                    #two seperate images of the surface
                    if boundary_count == 3:
                        return False

                prev_is_inside = curr_is_inside

        return True

    @staticmethod
    def create_from_file(surface_file):
        """ Creates a Surface object from a file exported from Imaris. Expects
            A file in the .npz format """

        surf = np.load(surface_file)

        xmax = surf[BBOX_MAX_X]
        xmin = surf[BBOX_MIN_X]

        ymax = surf[BBOX_MAX_Y]
        ymin = surf[BBOX_MIN_Y]

        zmax = surf[BBOX_MAX_Z]
        zmin = surf[BBOX_MIN_Z]

        nvox_x = surf[NUM_VOX_X]
        nvox_y = surf[NUM_VOX_Y]
        nvox_z = surf[NUM_VOX_Z]

        #nvox_ - 1 due to a quirk of how imaris decides to count voxels
        vdim_x = (xmax - xmin) / (nvox_x - 1)
        vdim_y = (ymax - ymin) / (nvox_y - 1)
        vdim_z = (zmax - zmin) / (nvox_z - 1)

        bbox = BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)

        vdim = (vdim_x, vdim_y, vdim_z)

        #imaris may export data with extraneous dimensions with no data
        vdata = np.squeeze(surf[VDATA])

        #if the surface is only availible at one time point we will squeeze too
        #far, so we need to add the time dimension back in
        if len(vdata.shape) == 3:
            vdata = np.expand_dims(vdata, axis=0)

        return Surface(vdata, bbox, vdim)
