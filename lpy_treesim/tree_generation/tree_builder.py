#!/usr/bin/env python3
from dataclasses import dataclass
import sys
from pathlib import Path
from lpy_treesim import ColorManager
import json
from openalea.lpy import Lsystem
from openalea.plantgl.all import *
from lpy_treesim.tree_generation.mesh_to_cylinders import add_cylinder_params_to_json, get_all_cylinder_params
from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention


import logging
import lpy_treesim.utils.logging_conf

logger = logging.getLogger(__name__)

BASE_LPY_PATH = Path(__file__).resolve().parents[1] / "base_lpy.lpy"

# Ensure repository root is discoverable for prototype imports
sys.path.insert(0, str(BASE_LPY_PATH.parents[0]))


class TreeBuilder:
    def __init__(
        self,
        tree_name: str,
        seed_value: int,
        semantic_label: bool = False,
        instance_label: bool = False,
        per_cylinder_label: bool = False,
    ):
        self.branch_hierarchy = {}
        self.color_manager = ColorManager()
        self.extern_vars = {
            "prototype_builder_path": f"lpy_treesim.examples.{tree_name}.{tree_name}_prototypes.build_basicwood_prototypes",
            "trunk_class_path": f"lpy_treesim.examples.{tree_name}.{tree_name}_prototypes.Trunk",
            "simulation_config_class_path": f"lpy_treesim.examples.{tree_name}.{tree_name}_simulation.{tree_name.upper()}SimulationConfig",
            "simulation_class_path": f"lpy_treesim.examples.{tree_name}.{tree_name}_simulation.{tree_name.upper()}Simulation",
            "color_manager": self.color_manager,
            "axiom_pitch": 0.0,
            "axiom_yaw": 0.0,
            "branch_hierarchy": self.branch_hierarchy,
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "per_cylinder_label": per_cylinder_label,
            "seed_value": seed_value,
        }
        # Enable batch mode before displaying anything

        self.__lsystem = Lsystem(str(BASE_LPY_PATH), self.extern_vars)
        
        return

    def lsystem(self) -> Lsystem:
        return self.__lsystem

    def generate_tree(self, b_interactive=True):
        lstring = self.__lsystem.axiom
        if b_interactive:
            Viewer.start()
        for iteration in range(self.__lsystem.derivationLength):
            lstring = self.__lsystem.derive(lstring, iteration, 1)
            self.__lsystem.plot(lstring)
            if b_interactive:
                scene =  self.__lsystem.sceneInterpretation(lstring)
                Viewer.display(scene)
                input("Press Enter to continue...")

        if b_interactive:
            Viewer.exit()
        print(lstring)
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
        for key, branch in self.branch_hierarchy.items():
            key = key.lower().strip()
            if "trunk" in key:
                trunk_dict = tree.new_trunk(root_stock=-1)
                mapping[key] = trunk_dict
                trunk_id = trunk_dict["id"]
                for child in branch:
                    child_key = child.name.lower().strip()
                    if "branch" in child_key:
                        branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=[])
                        mapping[child_key] = branch_dict
                    elif "spur" in child_key:
                        spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=[])
                        mapping[child_key] = spur_dict
                    else:
                        print(f"Unknown key {child_key}")
            elif "branch" in key:
                if key not in mapping:
                    raise ValueError(f"Child {key} should already be in mapping dictionary")

                branch_dict = mapping[key]
                parent_ids = tree.get_parent_id_list(branch_dict)
                parent_ids.append(branch_dict["id"])
                trunk_id = tree.get_trunk_id(branch_dict)

                for child in branch:
                    child_key = child.name.lower().strip()
                    if child_key in mapping:
                        raise ValueError(f"Child {key} is already in mapping dictionary, child of {key}")

                    if "branch" in child_key:
                        branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=parent_ids)
                        mapping[child_key] = branch_dict
                    elif "spur" in child_key:
                        spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=parent_ids)
                        mapping[child_key] = spur_dict
            elif "spur" in key:
                pass
            elif "root" in key:
                pass
            else:
                print(f"Unknown key {key}")
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
    
    def get_metadata(self, part_dict: list) -> dict:
        """Export metadata based on label settings. Includes hierarchy and L-Py vars."""
        export_dict = {
            "seed_value": int(self.extern_vars["seed_value"]),
            "semantic_label": self.extern_vars["semantic_label"],
            "instance_label": self.extern_vars["instance_label"],
            "per_cylinder_label": self.extern_vars["per_cylinder_label"],
            "axiom_pitch": float(self.extern_vars["axiom_pitch"]),
            "axiom_yaw": float(self.extern_vars["axiom_yaw"]),
        }
        # Hierarchy
        export_dict["hierarchy"] = self.export_hierarchy_dict()
        export_dict["branch_locations"] = self.export_branch_location_dict()
        color_data = self.color_manager.export_mapping_dict()
        export_dict["color_mapping"] = color_data
        # Label data
        if self.extern_vars["semantic_label"]:
            ...
        if self.extern_vars["instance_label"]:
            ...
        if self.extern_vars["per_cylinder_label"]:
            export_dict["cylinder_data"] = get_all_cylinder_params(part_dict, cylinder_metadata=color_data)
        return export_dict

    def export_metadata(self, part_dict: list, metadata_path: str) -> dict:
        """Export metadata based on label settings. Includes hierarchy and L-Py vars."""
        logger.info(f"Exporting metadata to {metadata_path}...")
        export_dict = self.get_metadata(part_dict)
        with open(metadata_path, "w") as f:
            json.dump(export_dict, f, indent=4)
        return export_dict
