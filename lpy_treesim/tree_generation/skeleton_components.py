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
from lpy_treesim.tree_generation.tree_structure import TreeStructure


class JunctionComponent:
    def __init__(self):
        self.t_along = 0.0
        self.theta_around = 0.0
        self.pt_attach = (0, 0, 0)
        self.vec_attach = (0, 0, 0)
        self.ang_attach = 0.0   # Angle from dot product at attach point
        self.radius = 0.0
        self.parent_name = ""
        self.child_name = ""

    def set_attach(self, parent, parent_tree_part: dict, child):
        """ Set the closest attachment point and the angle of the attachment"""
        # Writing this rather pedantically - project the point onto the line segment and find the closest
        centers_as_np = np.array(parent.centroids)
        pt_start = np.array(child.start_pt)
        vec_start = np.array(child.start_vec)

        t_best = 0.0
        d_best = 1e30
        pt_best = child.start_pt
        ang_attach = 0.0
        indx_best = 0
        radius_best = 0.0
        for indx in range(0, centers_as_np.shape[0] - 1):
            vec_v = centers_as_np[indx+1, :] - centers_as_np[indx, :]
            vec_v_len = np.linalg.norm(vec_v)
            if np.isclose(vec_v_len, 0.0):
                continue
            vec_w = pt_start - centers_as_np[indx, :]
            # Calculate the t along the line, clamped between 0 and 1
            t = np.dot(vec_w, vec_v) / np.dot(vec_v, vec_v)
            t = np.clip(t, 0.0, 1.0)

            pt_reconstruct = centers_as_np[indx, :] + vec_v * t
    
            dist = np.linalg.norm(pt_start - pt_reconstruct)
            if dist < d_best:
                indx_best = indx
                d_best = dist
                t_best = parent.t_values[indx] + t * (parent.t_values[indx+1] - parent.t_values[indx])
                pt_best = (pt_reconstruct[0], pt_reconstruct[1], pt_reconstruct[2])
                ang_attach = np.dot(vec_v / vec_v_len, vec_start)
                radius_best = (1 - t) * parent.radii[indx] + t * parent.radii[indx+1]

        self.t_along = float(t_best)
        self.pt_attach = (float(pt_best[0]), float(pt_best[1]), float(pt_best[2]))   
        self.ang_attach = float(360.0 * np.acos(ang_attach) / (2.0 * np.pi))
        self.radius = float(radius_best)  
        
        # Now do theta
        n_around = len(parent_tree_part["vertices"]) / centers_as_np.shape[0]
        n_around = int(n_around)
        # Just the vertices as indx
        mesh_vs_as_np = np.zeros((n_around+1, 3))
        mesh_vs_as_np[0:-1, :] = np.array(parent_tree_part["vertices"][indx_best * n_around:(indx_best+1) * n_around])
        mesh_vs_as_np[-1, :] = mesh_vs_as_np[0, :]
        # Project onto this ring
        theta_best = 0.0
        d_best = 1e30
        for indx in range(0, mesh_vs_as_np.shape[0] - 1):
            vec_v = mesh_vs_as_np[indx+1, :] - mesh_vs_as_np[indx, :]
            vec_v = vec_v / np.linalg.norm(vec_v)
            vec_w = pt_start - mesh_vs_as_np[indx, :]
            # Calculate the t along the line, clamped between 0 and 1
            t = np.dot(vec_w, vec_v)
            t = np.clip(t, 0.0, 1.0)

            pt_reconstruct = mesh_vs_as_np[indx, :] + vec_v * t
    
            dist = np.linalg.norm(pt_start - pt_reconstruct)
            if dist < d_best:
                d_best = dist
                theta_best = indx * (360.0 / n_around) + t
    
        # Convert to theta
        self.theta_around = float(theta_best)

    def create_dict(self) ->dict:
        ret_dict = {"t_along": self.t_along,
                    "theta_around": self.theta_around,
                    "pt_attach": self.pt_attach,
                    "vec_attach": self.vec_attach,
                    "ang_attach": self.ang_attach,
                    "radius": self.radius,
                    "parent_name": self.parent_name,
                    "child_name": self.child_name}
        return ret_dict

    def set_from_dict(self, in_dict: dict):
        self.t_along = in_dict["t_along"]
        self.theta_around = in_dict["theta_around"]
        self.pt_attach = in_dict["pt_attach"]
        self.vec_attach = in_dict["vec_attach"]
        self.ang_attach = in_dict["ang_attach"]
        self.radius = in_dict["radius"]
        self.child_name = in_dict["child_name"]
        self.parent_name = in_dict["parent_name"]


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
        centroid = np.mean(vs_as_np, axis=0)
        # Really annoying to cast to float, but otherwise json doesn't work
        self.centroids.append((float(centroid[0]), float(centroid[1]), float(centroid[2])))
        radius = float(np.linalg.norm(vs_as_np[0, :] - centroid[:]))
        self.radii.append(radius)

    def compute_t_values(self):
        """ Call AFTER all cylinders have been added"""
        # Really annoying to cast to float, but otherwise json doesn't work

        centers_as_np = np.array(self.centroids)
        self.start_pt = (float(centers_as_np[0, 0]), float(centers_as_np[0, 1]), float(centers_as_np[0, 2]))
        self.end_pt = (float(centers_as_np[-1, 0]), float(centers_as_np[-1, 1]), float(centers_as_np[-1, 2]))

        vec_to_first_pt = centers_as_np[1, :] - centers_as_np[0, :]
        len_vec = np.linalg.norm(vec_to_first_pt)
        if not np.isclose(len_vec, 0.0):
            vec_to_first_pt = vec_to_first_pt / len_vec
            self.start_vec = (float(vec_to_first_pt[0]), float(vec_to_first_pt[1]), float(vec_to_first_pt[2]))
        else:
            print(f"Warning, zero length vec {self.name}")

        dists = np.zeros(len(self.centroids))
        for indx in range(0, len(self.centroids) - 1):
            start_pt = centers_as_np[indx, :]
            end_pt = centers_as_np[indx + 1, :]
            dist = np.linalg.norm(end_pt - start_pt)
            dists[indx+1] = dist

        self.length = float(np.sum(dists))
        if self.length > 0.0:
            dists = dists / self.length
        self.t_values = []
        dist_sum = dists[0]
        for dist in dists[1:]:
            self.t_values.append(float(dist_sum))
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


def calculate_skeleton_junctions(tree: TreeStructure):
    """ Once the cylinders have been stitched together, create the junctions from them"""
    for junction_lists in (tree.trunk_junctions, tree.branch_junctions):
        for junction in junction_lists:
            parent_dict = tree.map_full_name_to_part[junction.parent_name]            
            child_dict = tree.map_full_name_to_part[junction.child_name]
            junction.set_attach(parent_dict["skel"], parent_dict["mesh"], child_dict["skel"])
