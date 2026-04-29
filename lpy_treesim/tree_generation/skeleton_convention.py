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

class SkeletonComponent:
    def __init__(self):
        self.centroid


    def new_trunk_cylinder(self, trunk_id: int):
        if trunk_id >= self.current_trunk_id:
            raise ValueError("Trying to add a trunk cylinder w/o having created the trunk id {trunk_id}")

        self.cyl_id += 1
        cyl_unique_id = self.cyl_id
        self.current_trunk_cyl_id[trunk_id] += 1
        cyl_trunk_id = self.current_trunk_cyl_id[trunk_id]
        trunk_name = TreeNamingConvention._trunk_name(self.current_trunk_id)

        # Add this cylinder id to the trunk dictionary list
        trunk_dict = self.part_list[TreeNamingConvention.part_names[TreeNamingConvention.TRUNK]][trunk_name]
        trunk_dict["components"].append[cyl_unique_id]
        cyl_unique_name = f"{trunk_dict['full_name']}_{TreeNamingConvention._cyl_name{cyl_unique_id}
        self.cyl_list[cyl_unique_id] = f"{trunk_dict['full_name']}_{cyl_unique_name}"
    @staticmethod
    def _cyl_name(cyl_id:int):
        cyl_name = f"CId-{cyl_id}"
        return cyl_name

