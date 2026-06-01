"""
Keeping skeleton data for tree structures

This module provides utilities for keeping skeletoal meta-data structure information. It both stores the skeletal
information and links mesh vertices/faces to skeleton components.

Because most tree skeletons are made out of cylinders or some sort of swept surface, and we want to tie the skeleton
to the mesh/point cloud geometry, the decision was made to keep the skeleton as a sequence of simplified cylinders
(a centroid with a vector and a radius) with the end points of the skeleton derived from that data

One skeleton component corresponds to a trunk or a branch in the naming_convention scheme

Data kept for the skeleton (if available)
- Skeleton is separated into a sequence of centroids + vectors
- At each centroid, the radius of the cylinder at that point
- t value (percentage along the skeleton)
- A coordinate system with one vector pointing along the cylinder and the second pointing in the direction of the seam
- mesh faces and vertices (or point cloud) that correspond to each segment

"""
import numpy as np
import shapely as shapely
from openalea.phenomenal.calibration.transformations import vector_product


class JunctionComponent:
    def __init__(self):
        self.t_along = 0.0
        self.theta_around = 0.0
        self.pt_attach = (0, 0, 0)
        self.vec_attach = (0, 0, 0)
        self.ang_attach = 0.0   # Dot product at attach point
        self.parent_name = ""
        self.child_name = ""

    def set_attach(self, parent: SkeletonComponent, parent_tree: dict, child: SkeletonComponent):
        """ Set the closest attachment point and the angle of the attachment"""
        # Writing this rather pedantically - project the point onto the line segment and find the closest
        centers_as_np = np.array(parent.centroids)
        line_in_shapely = shapely.linestrings(x=centers_as_np[:, 0], y=centers_as_np[:, 1], z=centers_as_np[:, 2])
        pt_project_in_shapely = shapely.Point(child.start_pt[0], child.start_pt[1], child.start_pt[2])
        n_pts_t = line_in_shapely.project(pt_project_in_shapely)
        # Convert from index to 0, 1
        self.t_along = n_pts_t / len(parent.centroids)
        pt_on_skeleton = line_in_shapely.interpolate(n_pts_t)
        self.pt_attach = (pt_on_skeleton[0], pt_on_skeleton[1], pt_on_skeleton[2])

        # This shouldn't be the case, but...
        indx = np.floor(n_pts_t)
        if indx > centers_as_np.shape[0] - 1:
            indx -= 1
        vec_parent = centers_as_np[indx + 1, :] - centers_as_np[indx, :]
        len_vec = np.linalg.norm(vec_parent)
        if not np.isclose(len_vec, 0.0):
            vec_parent = vec_parent / len_vec
        # Angle between trunk and branch (or branch and spur)
        self.ang_attach = np.dot(vec_parent, child.start_vec)

        # Now do theta
        n_around = len(parent["vertices"]) / centers_as_np.shape[0]
        # Just the vertices as indx
        mesh_vs_as_np = np.array(parent_tree["vertices"][indx * n_around:(indx+1) * n_around])
        # Project onto this ring
        pt_project_ring_in_shapely = shapely.linearrings(mesh_vs_as_np[:, 0], mesh_vs_as_np[:, 1], mesh_vs_as_np[:, 2])
        pt_on_ring = np.zeros((1, 3))
        radii = parent.radii[indx]
        for icoord in range(0, 3):
            pt_on_ring[icoord] = pt_on_skeleton[icoord] + child.start_vec[icoord] * radii
        n_ring_t = pt_project_ring_in_shapely.project(pt_on_ring)
        # Convert to theta
        self.theta_around = 360.0 * n_ring_t / n_around

    def create_dict(self) ->dict:
        ret_dict = {"t_along": self.t_along,
                    "theta_around": self.theta_around,
                    "pt_attach": self.pt_attach,
                    "vec_attach": self.pt_attach,
                    "ang_attach": self.ang_attach,
                    "child_name": self.child_name}
        return ret_dict

    def set_from_dict(self, in_dict: dict):
        self.t_along = in_dict["t_along"]
        self.theta_around = in_dict["theta_around"]
        self.pt_attach = in_dict["pt_attach"]
        self.vec_attach = in_dict["vec_attach"]
        self.vec_attach = in_dict["ang_attach"]
        self.child_name = in_dict["child_name"]


class SkeletonComponent:
    def __init__(self, name:str):
        self.name = name
        # Computed as cylinders are processed
        self.centroids = []
        self.radii = []
        # Set when parsing branch hierarchy
        self.start_pt = (0, 0, 0)
        self.start_vec = (1, 0, 0)
        self.end_pt = (0, 0, 0)
        self.child_junctions = []
        # Computed after cylinders are processed
        self.t_values = []
        self.length = 0.0

    def add_cylinder(self, vs: list):
        vs_as_np = np.array(vs)
        centroid = np.mean(vs_as_np, axis=1)
        self.centroids.append((centroid[0], centroid[1], centroid[2]))
        mid = len(vs) // 2
        radius = np.linalg.norm(vs_as_np[0, :] - vs_as_np[mid, :])
        self.radii.append(radius)

    def compute_t_values(self):
        """ Call AFTER all cylinders have been added"""
        centers_as_np = np.array(self.centroids)
        self.start_pt = (centers_as_np[0, 0], centers_as_np[0, 1], centers_as_np[0, 2])
        self.end_pt = (centers_as_np[-1, 0], centers_as_np[-1, 1], centers_as_np[-1, 2])

        vec_to_first_pt = centers_as_np[1, :] - centers_as_np[0, :]
        len_vec = np.linalg.norm(vec_to_first_pt)
        if not np.isclose(len_vec, 0.0):
            self.start_vec = vec_to_first_pt / len_vec
        else:
            print(f"Warning, zero length vec {self.name}")

        dists = np.zeros(len(self.centroids) - 1)
        for indx in range(0, len(self.centroids) - 1):
            start_pt = centers_as_np[indx, :]
            end_pt = centers_as_np[indx + 1, :]
            dist = np.linalg.norm(end_pt - start_pt)
            dists[indx+1] = dist

        self.length = np.sum(dists)
        if self.length > 0.0:
            dists = dists / self.length
        self.t_values = []
        dist_sum = dists[0]
        for dist in dists[1:]:
            self.t_values.append(dist_sum)
            dist_sum += dist

    def add_junction(self, child_component: JunctionComponent):
        self.child_junctions.append(child_component)

    def create_dict(self) ->dict:
        ret_dict = {}
        ret_dict["name"] = self.name
        ret_dict["centroids"] = self.centroids
        ret_dict["radii"] = self.radii
        ret_dict["start_pt"] = self.start_pt
        ret_dict["start_vec"] = self.start_vec
        ret_dict["end_pt"] = self.end_pt
        ret_dict["t_values"] = self.t_values
        ret_dict["length"] = self.length
        ret_dict["child_junctions"] = []
        for child in self.child_junctions:
            ret_dict["child_junctions"].append(child.create_dict())
        return ret_dict

    def set_from_dict(self, in_dict: dict):
        self.name = in_dict["name"]
        self.centroids = in_dict["centroids"]
        self.radii = in_dict["radii"]
        self.start_pt = in_dict["start_pt"]
        self.start_vec = in_dict["start_vec"]
        self.end_pt = in_dict["end_pt"]
        self.t_values = in_dict["t_values"]
        self.length = in_dict["length"]
        self.child_junctions = []
        for child_dict in in_dict["child_junctions"]:
            child = JunctionComponent()
            child.set_from_dict(child_dict)
            self.child_junctions.append(child)
