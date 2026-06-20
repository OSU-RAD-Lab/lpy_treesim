#!/usr/bin/env python3
import sys
from pathlib import Path
from lpy_treesim import ColorManager
import json
from openalea.lpy import Lsystem
from openalea.plantgl.all import *
from lpy_treesim.tree_generation.skeleton_components import SkeletonComponent, JunctionComponent
from lpy_treesim.tree_generation.tree_naming_convention import TreeNamingConvention
from lpy_treesim.tree_generation.tree_structure import TreeStructure
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicWood
import logging


class TreeBuilder:
    logger = logging.getLogger(__name__)
    b_init = False

    BASE_LPY_PATH = Path(__file__).resolve().parents[1] / "lpy_functions" / "base_lpy.lpy"

    def __init__(self,
                 tree_name: str,
                 seed_value: int,
                 interactive: bool):

        if not  TreeBuilder.b_init:
            # Ensure repository root is discoverable for prototype imports
            sys.path.insert(0, str(TreeBuilder.BASE_LPY_PATH.parents[0]))
            TreeBuilder.b_init = True

        # Store the branches as they're created
        self.branch_hierarchy = {}

        # For unique labeling of branch cylinders
        self.color_manager = ColorManager()

        # Show tree or not while building
        self.b_interactive = interactive

        # Where to find the source code and start values for the trunk
        self.extern_vars = {
            "prototype_builder_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.build_basicwood_prototypes",
            "trunk_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.Trunk",
            "simulation_config_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}SimulationConfig",
            "simulation_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}Simulation",
            "branch_hierarchy": self.branch_hierarchy,
            "color_manager": self.color_manager,
            "axiom_pitch": 0.0,
            "axiom_yaw": 0.0,
            "seed_value": seed_value,
            "interactive": self.b_interactive
        }
        # Enable batch mode before displaying anything
        self.__lsystem = Lsystem(str(TreeBuilder.BASE_LPY_PATH), self.extern_vars)

    def lsystem(self) -> Lsystem:
        return self.__lsystem

    @staticmethod
    def convert_vec3_to_tuple(vec3) -> tuple:
        return (float(vec3.x), float(vec3.y), float(vec3.z))

    @staticmethod
    def _replace_keyword(piece: str)->str:
        check_keywords = ["Support", "Bud", "Spur", "Trunk", "Branch"]
        for kw in check_keywords:
            if kw in piece:
                return "!" + kw + "!"
        # No keyword found - don't replace
        return "!" + piece + "!"

    @staticmethod
    def _take_out_class_refs(piece_in: str)->str:
        piece = piece_in
        while "<" in piece:
            piece_kw = ""
            if piece[0] == "<":
                split_break = piece.split(">")
                piece_kw = TreeBuilder._replace_keyword(split_break[0])
                piece = piece_kw + "".join(split_break[1:])
            else:
                split_break_left = piece.split("<")
                split_break_right = split_break_left[1].split(">")
                piece_kw = TreeBuilder._replace_keyword(split_break_right[0])
                piece = split_break_left[0] + piece_kw + "".join(split_break_right[1:])
                for indx in range(2, len(split_break_left)):
                    piece = piece + "<" + split_break_left[indx]
        return piece

    @staticmethod
    def make_string_readable(lstring: str):

        str_break_left_bracket = lstring.split('[')
        indent = 0
        for before_bracket in str_break_left_bracket:
            indent += 2

            b_has_end_bracket = False
            if len(before_bracket) > 0:
                b_has_end_bracket = before_bracket[-1] == "]"

            str_break_end_bracket = before_bracket.split("]")
            n_spaces = " " * indent
            print(f"\n{n_spaces}[", end="")
            for indx, up_to_end_bracket in enumerate(str_break_end_bracket):
                n_spaces = " " * indent
                piece = up_to_end_bracket.replace("WoodStart", "\n" + n_spaces + "WoodStart")
                piece = piece.replace("ParameterSet(type=", "PS(")
                piece = TreeBuilder._take_out_class_refs(piece)
                print(f"{piece}", end="")
                if indx < len(str_break_end_bracket) - 1 or b_has_end_bracket:
                    print(f"]")
                    indent -= 2

    def generate_tree(self):
        """ Actually build the lpy string
        @param b_interactive - do you want to have to hit a key every iteration?"""

        lstring = self.__lsystem.axiom

        if self.b_interactive:
            Viewer.start()

        for iteration in range(self.__lsystem.derivationLength):
            # One iteration - replace symbols
            lstring = self.__lsystem.derive(lstring, iteration, 1)

            print(f"Iteration {iteration}")
            self.make_string_readable(str(lstring))

            # DO NOT TAKE OUT THIS LINE - or everything will stop working
            # This calls all the code in the "Interpretation" block in base_lpy.py (the I() modules)
            interpreted_string = self.__lsystem.interpret(lstring)

            self.make_string_readable(str(interpreted_string))
            if self.b_interactive:
                scene =  self.__lsystem.sceneInterpretation(interpreted_string)
                Viewer.display(scene)
                input("Press Enter to continue...")

        if self.b_interactive:
            Viewer.exit()

        # String and scene (which has geometry)
        return lstring, self.__lsystem.sceneInterpretation(lstring)

    def _add_buds(self,
                  mapping_tree_structure: dict,
                  mapping_lpy: dict,
                  tree: TreeStructure,
                  parent_dict: dict,
                  parent_lpy: BasicWood,
                  bud_sites: list[BudSite]):
        new_junctions = []
        trunk_id = TreeNamingConvention.get_trunk_id(parent_dict)
        parent_ids = tree.get_parent_id_list(parent_dict)
        parent_name = parent_dict["name"]
        for bud in bud_sites:
            child_add_name = []
            if bud.branch_child:
                child_name = bud.branch_child.name
                branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=parent_ids)
                mapping_tree_structure[child_name] = branch_dict
                mapping_lpy[child_name] = bud.branch_child
                child_add_name.append(branch_dict["name"])
            if bud.spur_child:
                child_name = bud.spur_child.name
                spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=parent_ids)
                mapping_tree_structure[child_name] = spur_dict
                mapping_lpy[child_name] = bud.branch_spur
                child_add_name.append(spur_dict["name"])

            for child_name in child_add_name:
                junction = JunctionComponent()
                junction.parent_name = parent_name
                junction.child_name = child_name
                junction.t_along = bud.dist_along
                junction.radius = parent_lpy.growth.get_diameter(bud.dist_along)
                junction.pt_attach = bud.start_loc
                junction.ang_attach = bud.bud_angle_from_parent
                new_junctions.append(junction)

        return new_junctions

    def create_tree_structure(self) -> (TreeStructure, dict):
        tree = TreeStructure()

        mapping_tree_structure = {}
        mapping_lpy = {}
        # This is organized as name -> list of buds
        for key_orig, child_list in self.branch_hierarchy.items():
            key = key_orig.lower().strip()
            if "root" in key:
                # Root - start adding trunks
                root_dict = tree.new_root()
                for trunk in child_list:
                    trunk_dict = tree.new_trunk()
                    mapping_tree_structure[trunk.name] = trunk_dict
                    mapping_lpy[trunk.name] = trunk
            elif "bud" in key:
                continue
            else:
                branch_dict = mapping_tree_structure[key_orig]
                branch_lpy = mapping_lpy[key_orig]
                nj = self._add_buds(mapping_tree_structure=mapping_tree_structure,
                                    mapping_lpy=mapping_lpy,
                                    tree=tree,
                                    parent_dict=branch_dict,
                                    parent_lpy=branch_lpy,
                                    bud_sites=child_list)
                if "trunk" in key:
                    tree.trunk_junctions.extend(nj)
                else:
                    tree.branch_junctions.extend(nj)

        # Fill in remaining skeleton components
        for part_name, part_dict in mapping_tree_structure.items():
            lpy_part = mapping_lpy[part_name]
            part_dict["skel"] = SkeletonComponent(part_dict["name"])
            part_dict["skel"].start_pt = self.convert_vec3_to_tuple(lpy_part.location.start)
            part_dict["skel"].start_vec = self.convert_vec3_to_tuple(lpy_part.location.start_dir)
            part_dict["skel"].end_pt = self.convert_vec3_to_tuple(lpy_part.location.end)
        return tree, mapping_tree_structure

    def export_hierarchy_dict(self) -> dict:
        named_hierarchy = {}
        for key, branch in self.branch_hierarchy.items():
            key = key.lower().strip()
            named_hierarchy[key] = []
            for child in branch:
                named_hierarchy[key].append(child.name.lower().strip())
        return named_hierarchy

    def export_branch_location_dict(self) -> dict:
        named_hierarchy = {}
        for key, branch in self.branch_hierarchy.items():
            for child in branch:
                child_name = child.name.lower().strip()
                if "bud" in child_name:
                    named_hierarchy[child_name] = {"start": self.convert_vec3_to_tuple(child.start_loc)}
                else:
                    named_hierarchy[child_name] = {"start": self.convert_vec3_to_tuple(child.location.start),
                                                   "end": self.convert_vec3_to_tuple(child.location.end)}
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
