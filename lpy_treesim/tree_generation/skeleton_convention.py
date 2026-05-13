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


class JunctionComponent:
    def __init__(self):
        self.t_along = 0.0
        self.theta_around = 0.0
        self.pt_attach = (0, 0, 0)
        self.ang_attach = (0, 0, 0)
        self.child_name = ""

    def create_json(self) ->dict:
        ret_dict = {}
        ret_dict["t_along"] = self.t_along
        ret_dict["theta_around"] = self.theta_around
        ret_dict["pt_attach"] = self.pt_attach
        ret_dict["ang_attach"] = self.ang_attach
        ret_dict["child_name"] = self.child_name
        return ret_dict

    def set_from_dict(self, in_dict: dict):
        self.t_along = in_dict["t_along"]
        self.theta_around = in_dict["theta_around"]
        self.pt_attach = in_dict["pt_attach"]
        self.ang_attach = in_dict["ang_attach"]
        self.child_name = in_dict["child_name"]


class SkeletonComponent:
    def __init__(self, name:str):
        self.name = str
        # Computed as cylinders are processed
        self.centroids = []
        self.radii = []
        # Set when parsing branch hierarchy
        self.start_pt = (0, 0, 0)
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
        start_pt = np.array(self.start_pt)
        end_pt = centers_as_np[0, :]
        dists = np.zeros(len(self.centroids) + 1)
        for indx in range(0, len(self.centroids)):
            dist = np.linalg.norm(end_pt - start_pt)
            dists[indx] = dist
            start_pt = centers_as_np[indx]
            if indx == len(self.centroids) - 1:
                end_pt = np.array(self.end_pt)
            else:
                end_pt = centers_as_np[indx+1]
        dist = np.linalg.norm(end_pt - start_pt)
        dists[-1] = dist
        self.length = np.sum(dists)
        if self.length > 0.0:
            dists = dists / self.length
        self.t_values = []
        for indx in range(0, len(self.centroids)):
            self.t_values.append(dists[indx])


    def add_junction(self, child_component: JunctionComponent):
        self.child_junctions.append(child_component)

    def create_json(self) ->dict:
        ret_dict = {}
        ret_dict["name"] = self.name
        ret_dict["centroids"] = self.centroids
        ret_dict["radii"] = self.radii
        ret_dict["start_pt"] = self.start_pt
        ret_dict["end_pt"] = self.end_pt
        ret_dict["t_values"] = self.t_values
        ret_dict["length"] = self.length
        ret_dict["child_junctions"] = []
        for child in self.child_junctions:
            ret_dict["child_junctions"].append(child.create_json())
        return ret_dict

    def set_from_dict(self, in_dict: dict):
        self.name = in_dict["name"]
        self.centroids = in_dict["centroids"]
        self.radii = in_dict["radii"]
        self.start_pt = in_dict["start_pt"]
        self.end_pt = in_dict["end_pt"]
        self.t_values = in_dict["t_values"]
        self.length = in_dict["length"]
        self.child_junctions = []
        for child_dict in in_dict["child_junctions"]:
            child = JunctionComponent()
            child.set_from_dict(child_dict)
            self.child_junctions.append(child)
