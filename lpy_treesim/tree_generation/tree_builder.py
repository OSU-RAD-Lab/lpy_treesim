#!/usr/bin/env python3
import sys
from pathlib import Path
from lpy_treesim import ColorManager
import json
from openalea.lpy import Lsystem
from openalea.plantgl.all import *
from lpy_treesim.tree_generation.skeleton_convention import SkeletonComponent, JunctionComponent
from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention
import logging


class TreeBuilder:
    logger = logging.getLogger(__name__)

    BASE_LPY_PATH = Path(__file__).resolve().parents[1] / "tree_models" / "base_tree" / "base_lpy.lpy"

    def __init__(
        self,
        tree_name: str,
        seed_value: int,
    ):
        # Ensure repository root is discoverable for prototype imports
        sys.path.insert(0, str(TreeBuilder.BASE_LPY_PATH.parents[0]))

        self.branch_hierarchy = {}
        self.color_manager = ColorManager()
        self.extern_vars = {
            "prototype_builder_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.build_basicwood_prototypes",
            "trunk_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.Trunk",
            "simulation_config_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}SimulationConfig",
            "simulation_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}Simulation",
            "color_manager": self.color_manager,
            "axiom_pitch": 0.0,
            "axiom_yaw": 0.0,
            "branch_hierarchy": self.branch_hierarchy,
            "seed_value": seed_value,
        }
        # Enable batch mode before displaying anything

        self.__lsystem = Lsystem(str(TreeBuilder.BASE_LPY_PATH), self.extern_vars)

    def lsystem(self) -> Lsystem:
        return self.__lsystem

    def generate_tree(self, b_interactive=True):
        lstring = self.__lsystem.axiom
        if b_interactive:
            Viewer.start()
        for iteration in range(self.__lsystem.derivationLength):
            lstring = self.__lsystem.derive(lstring, iteration, 1)
            # DO NOT TAKE OUT THIS LINE - or everything will stop working
            self.__lsystem.plot(lstring)
            if b_interactive:
                scene =  self.__lsystem.sceneInterpretation(lstring)
                Viewer.display(scene)
                input("Press Enter to continue...")

        if b_interactive:
            Viewer.exit()
        return lstring, self.__lsystem.sceneInterpretation(lstring)
    
    def export_hierarchy_dict(self) -> dict:
        named_hierarchy = {}
        for key, branch in self.branch_hierarchy.items():
            key = key.lower().strip()
            named_hierarchy[key] = []
            for child in branch:
                named_hierarchy[key].append(child.name.lower().strip())
        return named_hierarchy

    def create_tree_structure(self) -> (TreeNamingConvention, dict):
        tree = TreeNamingConvention()

        mapping = {}
        for key_orig, branch in self.branch_hierarchy.items():
            key = key_orig.lower().strip()
            if "root" in key:
                root_dict = tree.new_root()
                mapping[key_orig] = root_dict
            elif "trunk" in key:
                trunk_dict = tree.new_trunk()
                mapping[key_orig] = trunk_dict
                trunk_id = trunk_dict["id"]
                for child in branch:
                    junction = JunctionComponent()
                    junction.parent_name = trunk_dict["name"]
                    tree.trunk_junctions.append(junction)

                    child_key = child.name.lower().strip()
                    if "branch" in child_key:
                        branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=[])
                        mapping[child.name] = branch_dict
                        junction.child_name = branch_dict["name"]
                    elif "spur" in child_key:
                        spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=[])
                        mapping[child.name] = spur_dict
                        junction.child_name = spur_dict["name"]
                    else:
                        print(f"Unknown key {child_key}")
            elif "branch" in key:
                if key_orig not in mapping:
                    raise ValueError(f"Child {key} should already be in mapping dictionary")

                branch_dict = mapping[key_orig]
                parent_ids = tree.get_parent_id_list(branch_dict)
                parent_ids.append(branch_dict["id"])
                trunk_id = tree.get_trunk_id(branch_dict)

                junction = JunctionComponent()
                junction.parent_name = branch_dict["name"]
                tree.branch_junctions.append(junction)

                for child in branch:
                    child_key = child.name.lower().strip()
                    if child.name in mapping:
                        raise ValueError(f"Child {child.name} is already in mapping dictionary, child of {key_orig}")

                    if "branch" in child_key:
                        branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=parent_ids)
                        mapping[child.name] = branch_dict
                        junction.child_name = branch_dict["name"]
                    elif "spur" in child_key:
                        spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=parent_ids)
                        mapping[child.name] = spur_dict
                        junction.child_name = spur_dict["name"]
            elif "spur" in key:
                pass
            else:
                print(f"Unknown key {key}")

        # TODO: Extract out angle and t for junction
        for key, branch in self.branch_hierarchy.items():
            for child in branch:
                child_name = child.name
                part_dict = mapping[child_name]
                part_dict["skel"] = SkeletonComponent(part_dict["name"])
                part_dict["skel"].start_pt = self.convert_vec3_to_tuple(child.location.start)
                part_dict["skel"].end_pt = self.convert_vec3_to_tuple(child.location.end)

        # Do this after the skeleton parts have been created
        for junc in tree.trunk_junctions:
            trunk_part = tree.part_list[TreeNamingConvention._trunk_key()][junc.parent_name]
            trunk_part["skel"].add_junction(junc)

        for junc in tree.branch_junctions:
            branch_part = tree.part_list[TreeNamingConvention._branch_key()][junc.parent_name]
            branch_part["skel"].add_junction(junc)

        return tree, mapping

    @staticmethod
    def convert_vec3_to_tuple(vec3) -> tuple:
        return (float(vec3.x), float(vec3.y), float(vec3.z))
    
    def export_branch_location_dict(self) -> dict:
        named_hierarchy = {}
        for key, branch in self.branch_hierarchy.items():
            for child in branch:
                child_name = child.name.lower().strip()
                named_hierarchy[child_name] = {"start": self.convert_vec3_to_tuple(child.location.start), "end": self.convert_vec3_to_tuple(child.location.end)}
        return named_hierarchy
    
    def get_metadata(self) -> dict:
        """Export metadata based on label settings. Includes hierarchy and L-Py vars."""
        export_dict = {
            "seed_value": int(self.extern_vars["seed_value"]),
            "axiom_pitch": float(self.extern_vars["axiom_pitch"]),
            "axiom_yaw": float(self.extern_vars["axiom_yaw"]),
        }
        # Hierarchy
        export_dict["hierarchy"] = self.export_hierarchy_dict()
        export_dict["branch_locations"] = self.export_branch_location_dict()
        return export_dict

    def export_metadata(self, metadata_path: str) -> dict:
        """Export metadata based on label settings. Includes hierarchy and L-Py vars."""
        TreeBuilder.logger.info(f"Exporting metadata to {metadata_path}...")
        export_dict = self.get_metadata()
        with open(metadata_path, "w") as f:
            json.dump(export_dict, f, indent=4)
        return export_dict
