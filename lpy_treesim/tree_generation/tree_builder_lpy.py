#supress file print statements
#original_print = print
'''
import builtins

original_print = builtins.print

def smart_print(*args, **kwargs):
    # Intercept and silence the specific L-System geometry and mesh spam
    if args and isinstance(args[0], str):
        if "Bad growth length" in args[0] or "has no mesh, removing" in args[0]:
            return
    original_print(*args, **kwargs)

# Override the global print function
builtins.print = smart_print
'''

#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import trimesh
import numpy as np
import pandas as pd
from pathlib import Path

from lpy_treesim.utils.color_manager import ColorManager
import json
from openalea.lpy import Lsystem
from openalea.plantgl.all import *
from lpy_treesim.tree_generation.skeleton_components import SkeletonComponent, JunctionComponent
from lpy_treesim.tree_generation.tree_naming_convention import TreeNamingConvention
from lpy_treesim.tree_generation.tree_structure import TreeStructure
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicWood
from lpy_treesim.tree_generation.tree_to_usd import create_mesh_usd
from lpy_treesim.tree_generation.lpy_scene_to_mesh import plant_gl_scene_to_vertices_and_faces, stitch_cylinders, write_mesh
from lpy_treesim.textures.generate_texture import make_texture_set, make_uv_texture

import logging


class TreeBuilder:
    logger = logging.getLogger(__name__)
    b_init_sys_path = False

    # Where to find base_lpy.lpy, which is the one and only lpy file that defines the string generation
    BASE_LPY_PATH = Path(__file__).resolve().parents[1] / "lpy_functions" / "base_lpy.lpy"

    def __init__(self,
                 tree_name: str,
                 seed_value: int,
                 args):

        if not TreeBuilder.b_init_sys_path:
            # Ensure repository root is discoverable for prototype imports
            sys.path.insert(0, str(TreeBuilder.BASE_LPY_PATH.parents[0]))
            TreeBuilder.b_init_sys_path = True

        # Store the branches as they're created
        self.branch_hierarchy = {}
        self.map_name_instance = {}

        # For unique labeling of branch cylinders
        self.color_manager = ColorManager()

        # Show tree or not while building
        self.b_interactive = args.interactive
        #self.b_interactive = interactive

        # args set in make_n_trees
        self.args = args

        # Where to find the source code and start values for the trunk
        # See extern_variables in base_lpy.lpy
        self.extern_vars = {
            "prototype_builder_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.build_basicwood_prototypes",
            "trunk_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_prototypes.Trunk",
            "simulation_config_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}SimulationConfig",
            "simulation_class_path": f"lpy_treesim.tree_models.{tree_name}.{tree_name}_simulation.{tree_name.upper()}Simulation",
            "branch_hierarchy": self.branch_hierarchy,
            "map_name_instance": self.map_name_instance,
            "color_manager": self.color_manager,
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
        """ For making strings readable - converts pointers to strings"""
        check_keywords = ["Support", "Bud", "Spur", "Trunk", "Branch"]
        for kw in check_keywords:
            if kw in piece:
                return "!" + kw + "!"
        # No keyword found - don't replace
        return "!" + piece + "!"

    @staticmethod
    def _take_out_class_refs(piece_in: str)->str:
        """ For making strings readable - converts <> instances to strings"""
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
        """ A not very good bit of code for indenting strings by brackets []
        """
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

    def check_tree(self):
        found_buds = []
        found_branches = []
        for branch_key, branch_buds in self.branch_hierarchy.items():
            if "root" in branch_key:
                continue
            if not branch_key in self.map_name_instance:
                print(f"Checking tree: No branch {branch_key} in map instance")
                continue
            if "bud" in branch_key:
                continue
            branch = self.map_name_instance[branch_key]
            for bud in branch_buds:
                found_buds.append(bud.name)
                if bud.branch_child:
                    found_branches.append(bud.branch_child.name)
                if bud.spur_child:
                    found_branches.append(bud.spur_child.name)
                if bud.dist_along > branch.growth.length:
                    print(f"Problem: {branch.name}, {bud.name}")
        for bud_name in found_buds:
            if not bud_name in self.branch_hierarchy:
                print(f"Problem: {bud_name} is alive but not in hierarchy")
        for branch_name in found_branches:
            if not branch_name in self.branch_hierarchy:
                print(f"Problem: {branch_name} is alive but not in hierarchy")

    def check_string(self, lstring: str):
        find_wood = lstring.split("WoodStart")
        for string_bit in find_wood[1:]:
            if not string_bit[0] == "(":
                continue
            name = string_bit[1:].split(")")[0]
            if not name in self.branch_hierarchy:
                print(f"Could not find name {name} in branch hierarchy")
        find_spur = lstring.split("SpurStart")
        for string_bit in find_spur[1:]:
            if not string_bit[0] == "(":
                continue
            name = string_bit[1:].split(")")[0]
            if not name in self.branch_hierarchy:
                print(f"Could not find name {name} in branch hierarchy")

    def create_cylinder_mark(self,
                             normal_vector:np.ndarray,
                             location:tuple, 
                             radius:float = 0.05, 
                             color: list = [255, 0, 0, 255], 
                             height=0.005):
        
        #mag = np.linalg.norm(vector)
        # rotation that maps cylinder z-axis [0,0,1] to the desired direction
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], normal_vector)
        # build 4x4 transform: rotation + translation to midpoint
        transform[:3, 3] = location
        marker = trimesh.creation.cylinder(radius=radius, height=height, transform=transform)
        marker.visual.face_colors = color
        return marker


    def generate_tree(self, naming, index, radii, name_radii, stage_context):
        """ Actually build the lpy string
        @param b_interactive - do you want to have to hit a key every iteration?"""

        # First string - see base_lpy.lpy axiom module
        frozen_lstring = self.__lsystem.axiom
        lstring = self.__lsystem.axiom
        usd_path = "no_usd_path"

        if self.b_interactive:
            """Suppose to bring up a window. Whether or not it does depends on OS"""
            Viewer.start() # type: ignore
            
        # Iterate, replacing modules with new ones every iteration
        #  Roughly 28 iterations per year, 3-5 years (depending on SimulationConfig parameters)
        b_check_string = False

        year = 0
        snapshot_start = False
        snapshot_iteration = 0
        iteration = 0
        SNAPSHOT_ITER_TO_GENERATE = 3

        num_iter_per_year = self.__lsystem.simulation_config.num_iter_per_year

        while iteration <= self.__lsystem.derivationLength:
        #for iteration in range(self.__lsystem.derivationLength):

            #print(f"Iteration {iteration}")
            # print(f"Lpy Iteration {self.__lsystem.context().getIterationNb()}")

            if b_check_string:
                self.check_string(str(lstring))
                b_check_string = False

            # One iteration - replace symbols
            #self.make_string_readable(str(lstring))

            if (iteration % (num_iter_per_year)) == (num_iter_per_year - 1):  #and iteration != 0:
                if iteration < self.__lsystem.derivationLength - 4:
                    print("Deriving lstring copy")
                    snapshot_start = True
                    frozen_lstring = self.__lsystem.derive(lstring, iteration, 1)
                    # print(f"Lpy Iteration {self.__lsystem.context().getIterationNb()}")
                    iteration += 1
                    frozen_iteration = iteration
                    # print("Deriving current lstring")
                    lstring = self.__lsystem.derive(lstring, iteration, 1)
   

            elif snapshot_iteration == (SNAPSHOT_ITER_TO_GENERATE + 2): #change back to 2
                snapshot_start = False
                iteration = frozen_iteration
                lstring = self.__lsystem.derive(frozen_lstring, iteration, 1)

                # print(f"Frozen lstring derived on {self.__lsystem.context().getIterationNb()}")
                
            else:
                lstring = self.__lsystem.derive(lstring, iteration, 1)
                #print("lstring derived")

            if "%" in str(lstring):
                print("PRUNING cuts found in string")
                self.check_tree()
                b_check_string = True

            # DO NOT TAKE OUT THIS LINE - or everything will stop working
            # This calls all the code in the "Interpretation" block in base_lpy.py (the I() modules)
            interpreted_string = self.__lsystem.interpret(lstring)
            
            #self.make_string_readable(str(interpreted_string))
            if self.b_interactive:
                scene =  self.__lsystem.sceneInterpretation(lstring)
                Viewer.display(scene) # type: ignore
                input("Press Enter to continue...")

            if snapshot_iteration == SNAPSHOT_ITER_TO_GENERATE or iteration == (self.__lsystem.derivationLength): # Change back to 1
                year += 1
                print(f'Generating Tree on Iteration {iteration}')
                # String and scene (which has geometry)
                #   This string will have all the F() modules, which are the ones that actually produce geometry
                self.check_string(str(lstring))
                self.check_string(str(lstring))

                # Converts the scene to our tree structure.
                #   Mapping maps the unique ids from the lstring into our tree structure
                #   This ensures the branches etc are numbered sequentially
                scene = self.__lsystem.sceneInterpretation(lstring)
                tree, mapping_lpy, mapping_tree_structure = self.create_tree_structure()

                # Adds to each tree component the mesh cylinders created by lpy
                bud_sites = plant_gl_scene_to_vertices_and_faces(scene,
                                                                mapping_tree_structure=mapping_tree_structure,
                                                                color_mapping=self.color_manager)

                # Now stitch together all of the mesh components into tubes instead of discrete cylinders
                # Also adds colors and texture coordinates
                color_to_part, keys_to_remove = stitch_cylinders(tree=tree)
                # Some newly created branch parts do not have any meshes associated with them
                #for key in keys_to_remove:
                    #tree.remove_key(key)

                # Now that the cylinders/skeleton have been processed, build the junctions
                #calculate_skeleton_junctions(tree=tree)
                
                # Write out mesh file formats
                if self.args.ply or self.args.obj:
                    mesh_path = str(self.args.output_dir / naming.mesh_filename(index, file_type="")) + "_year" + str(year)
                    uv_name = str(mesh_path) + "_uv.png"
                    make_uv_texture(uv_name)
                    write_mesh(tree=tree, fname=mesh_path, bud_sites=bud_sites, image_name=uv_name)

                if stage_context is not [] and self.args.usda:
                    # Where the usd files are stored
                    usd_path = str(os.path.join(str(self.args.stage_dir), naming.usd_filename(index))) + str(year)
                    uv_name = str(self.args.stage_dir ) + "/textures/mesh_uv.png"
                    make_uv_texture(uv_name)
                    for b_use_uv in [True, False]:
                        create_mesh_usd(stage_context, 
                                        world_path=str(self.args.stage_dir), 
                                        in_tree_name=naming._prefix(index), 
                                        tree=tree, 
                                        radii=radii, name_radii=name_radii,
                                        b_use_uv=b_use_uv)
                del scene


                # Create indicators, from marked locations, on the tree after the tree has been generated

                # Create path for marked locations file and load generated tree mesh
                # Added this code. This gets the base name and appends the file to _vc.obj so trimesh will find the file
                mesh_name = naming.mesh_filename(index, file_type="") + "_year" + str(year) + "_vc.obj"
                mod_path = str(self.args.output_dir / mesh_name)
                mesh_existing = trimesh.load(mod_path)
                mesh_existing.visual = mesh_existing.visual.to_texture()
                mesh_existing.visual.material.alphaMode = "OPAQUE"

                scene = trimesh.Scene()
                scene.add_geometry(mesh_existing, node_name="base_tree")

                df = pd.read_csv(str(Path(__file__).resolve().parents[1] / "tie_prune" / "pruning_algo" / "marked_locations.csv"))

                df = df[df['Year'] == year]
                
                for n, row in enumerate(df.itertuples(index=False)):
                    # Grab coordinates as numpy arrays so we can do math on them
                    base_location = np.array([row.Loc_x, row.Loc_y, row.Loc_z], dtype=float)
                    start_coord = np.array([row.Norm_x1, row.Norm_y1, row.Norm_z1], dtype=float)
                    end_coord = np.array([row.Norm_x2, row.Norm_y2, row.Norm_z2], dtype=float)
                    
                    # Calculate the direction the branch is growing outward
                    branch_vector = end_coord - start_coord
                    branch_length = np.linalg.norm(branch_vector)
                    
                    # Slide the marker 2cm up the branch to clear the trunk
                    offset_distance = 0.005 
                    if branch_length > 0:
                        branch_direction = branch_vector / branch_length
                        location = tuple(base_location + (branch_direction * offset_distance))
                    else:
                        location = tuple(base_location)
                    
                    
                    normal_vector = branch_vector #np.array([1.0, 0.0, 0.0], dtype=float)
                
                    # Grab the single values
                    item_type = row.Type
                    
                    # Disc Sizing 
                    # Slightly wider than the branch so it's visible, but not massive
                    radius = row.Radius + 0.03  
                    # Trying to make the disc thinner
                    disc_height = 0.015
                    TRANSPARENCY = 120
                    # Marker radius and height are explicitly passed here
                    # Changed flag_for_no_replace to primary_without_replacement
                    if item_type == "bud_spacing":
                        marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                        marker.visual.face_colors = [0, 255, 0, 30]
                        marker.apply_translation(location)

                    elif item_type == "primary_to_prune":
                        # RED: Tied primary branch matched with a replacement
                        marker = self.create_cylinder_mark(normal_vector, location, radius=radius, height=disc_height, color=[255, 0, 0, TRANSPARENCY])
                    elif item_type == "primary_without_replacement":
                        # YELLOW: Tied primary branch with no available replacement
                        marker = self.create_cylinder_mark(normal_vector, location, radius=radius, height=disc_height, color=[255, 255, 0, TRANSPARENCY])
                    # TODO
                    elif item_type == "flag_for_replace":
                        # BLUE: Untied primary branch acting as the replacement
                        marker = self.create_cylinder_mark(normal_vector, location, radius=radius, height=disc_height, color=[0, 0, 255, TRANSPARENCY])

                    elif item_type == "vigor":
                        marker = self.create_cylinder_mark(normal_vector, location, radius=radius, height=disc_height, color=[255, 0, 0, TRANSPARENCY])
                    elif item_type == "canopy":
                        marker = self.create_cylinder_mark(normal_vector, location, radius=radius, height=disc_height, color=[0, 0, 255, TRANSPARENCY])

                    marker.visual = marker.visual.to_texture()
                    marker.visual.material.alphaMode = "BLEND"
                    scene.add_geometry(marker, node_name=f"marker_{n}")
                
                
                # Export as a separate file (e.g., marked_location_iter_29.obj)
                marked_file_name = "marked_locations_year" + str(year) + ".obj"
                out_name = self.args.output_dir / marked_file_name
                scene.export(out_name)
                print(f"Exported year file: {out_name}")

                if self.args.meta_data:
                    import json
                    metadata_path = self.args.output_dir / str(year) + "year_" + str(naming.metadata_filename(index))
                    meta_data = self.get_metadata()
                    meta_data["tree"] = tree.create_dict()
                    # meta_data["tree_mapping"] = mapping
                    meta_data["color_mapping"] = color_to_part
                    with open(metadata_path, "w") as f:
                        json.dump(meta_data, f, indent=4)
                else:
                    metadata_path = None
            
                    
            if snapshot_start:
                snapshot_iteration += 1
            else:
                snapshot_iteration = 0
                
            iteration += 1


        if self.b_interactive:
            Viewer.exit() # type: ignore
        return usd_path, mesh_path, metadata_path

    def _add_junctions(self,
                       mapping_tree_structure: dict,
                       parent_dict: dict,
                       parent_lpy: BasicWood,
                       bud_sites: list[BudSite]):
        """ Get all the junctions for this branch. Fills in the junction information"""
        new_junctions = []
        for bud_indx, bud in enumerate(bud_sites):
            new_junction = JunctionComponent()
            new_junction.t_along = bud.dist_along / parent_lpy.growth.length
            new_junction.theta_around = bud.bud_angle_around
            new_junction.pt_attach = self.convert_vec3_to_tuple(bud.start_loc)
            if bud_indx < len(bud_sites) - 1:
                vec_parent_dir = bud_sites[bud_indx + 1].start_loc - bud.start_loc
                vec_parent_dir /= np.linalg.norm(vec_parent_dir)
            else:
                vec_parent_dir = parent_lpy.location.end_dir
            new_junction.vec_along_parent = self.convert_vec3_to_tuple(vec_parent_dir)
            new_junction.vec_branch = self.convert_vec3_to_tuple(bud.start_dir)
            new_junction.ang_attach = bud.bud_angle_from_parent
            new_junction.radius = parent_lpy.growth.get_diameter(bud.dist_along)
            new_junction.parent_name = parent_dict["full_name"]
            if bud.bud_state is BudSite.BudType.PRUNED:
                new_junction.type = JunctionComponent.JunctionType.PRUNED
            if bud.bud_state is BudSite.BudType.DORMANT:
                new_junction.type = JunctionComponent.JunctionType.BUD
            if bud.branch_child:
                lpy_child_name = bud.branch_child.name
                child_dict = mapping_tree_structure[lpy_child_name]
                tree_child_name = child_dict["name"]
                new_junction.child_branch_name = tree_child_name
                new_junction.type = JunctionComponent.JunctionType.BRANCH
            if bud.spur_child:
                lpy_child_name = bud.spur_child.name
                child_dict = mapping_tree_structure[lpy_child_name]
                tree_child_name = child_dict["name"]
                new_junction.child_spur_name = tree_child_name
                if bud.branch_child:
                    new_junction.type = JunctionComponent.JunctionType.BRANCH_AND_SPUR
                else:
                    new_junction.type = JunctionComponent.JunctionType.BRANCH

            new_junctions.append(new_junction)

        return new_junctions

    def _add_buds(self,
                  mapping_tree_structure: dict,
                  mapping_lpy: dict,
                  tree: TreeStructure,
                  parent_dict: dict,
                  bud_sites: list[BudSite]):
        """ Turning branch_hierarchy into TreeStructure data structure
        This helper method loops over all the bud sites on a branch and adds the bud sites' branch/spur children
        to the parent_dict structure"""
        trunk_id = TreeNamingConvention.get_trunk_id(parent_dict)
        parent_ids = tree.get_parent_id_list(parent_dict)
        for bud in bud_sites:
            if bud.branch_child:
                child_lpy_name: str = bud.branch_child.name
                branch_dict = tree.new_branch(trunk_id=trunk_id, parent_ids=parent_ids)
                mapping_tree_structure[child_lpy_name] = branch_dict
                mapping_lpy[child_lpy_name] = bud.branch_child
            if bud.spur_child:
                child_lpy_name: str = bud.spur_child.name
                spur_dict = tree.new_spur(trunk_id=trunk_id, parent_and_branch_ids=parent_ids)
                mapping_tree_structure[child_lpy_name] = spur_dict
                mapping_lpy[child_lpy_name] = bud.spur_child

    def _add_skeleton(self,
                      mapping_tree_structure: dict,
                      mapping_lpy: dict,
                      branch_dict: dict,
                      branch_lpy: BasicWood,
                      bud_sites: list[BudSite]):
        """ Adds the skeleton components to the tree structure"""
        skel = SkeletonComponent(branch_dict["name"])
        skel.start_pt = self.convert_vec3_to_tuple(branch_lpy.location.start)
        skel.end_pt = self.convert_vec3_to_tuple(branch_lpy.location.end)
        skel.length = branch_lpy.growth.length

        skel.centroids.append(self.convert_vec3_to_tuple(branch_lpy.location.start))
        skel.radii.append(branch_lpy.growth.get_diameter(0.0))
        skel.t_values.append(0.0)
        for bud in bud_sites:
            skel.centroids.append(self.convert_vec3_to_tuple(bud.start_loc))
            skel.radii.append(branch_lpy.growth.get_diameter(bud.dist_along))
            skel.t_values.append(bud.dist_along / branch_lpy.growth.length)

        l = 0.0
        for indx in range(0, 3):
            l += (skel.centroids[-1][indx] - branch_lpy.location.end[indx]) ** 2

        if not np.isclose(l, 0.0):
            skel.centroids.append(self.convert_vec3_to_tuple(branch_lpy.location.end))
            skel.radii.append(branch_lpy.growth.get_diameter(branch_lpy.growth.length))
            skel.t_values.append(1.0)

        skel.child_junctions = self._add_junctions(mapping_tree_structure=mapping_tree_structure,
                                                   parent_dict=branch_dict,
                                                   parent_lpy=branch_lpy,
                                                   bud_sites=bud_sites)
        branch_dict["skel"] = skel

    def create_tree_structure(self) -> tuple[TreeStructure, dict, dict]:
        """ Loop over the branch structure and make one Tree component for each structure.
        Returns the tree structure and a mapping from the lpy names to the new tree structures"""
        tree = TreeStructure()

        # Map lpy names to tree parts
        mapping_tree_structure = {}
        # Map lpy names to Tree structure pointers
        mapping_lpy = {}
        # Each item in the branch_hierarchy dictionary is organized as name -> list of buds
        # This is relying on the fact that looping over the dictionary will happen in the order that the objects
        #  were created (eg root, trunk, primary branches)
        for key_orig, child_list in self.branch_hierarchy.items():
            key = key_orig.lower().strip()
            if "root" in key:
                # Root - start adding trunks
                root_dict = tree.new_root()
                for trunk_lpy in child_list:
                    trunk_dict = tree.new_trunk()
                    mapping_tree_structure[trunk_lpy.name] = trunk_dict
                    mapping_lpy[trunk_lpy.name] = trunk_lpy
            elif "bud" in key:
                # Skipping these because buds will be handled when processing each branch/trunk
                continue
            else:
                # The dictionary elements are created when we process the parent, so we just have to get
                # them back
                branch_dict = mapping_tree_structure[key_orig]
                branch_lpy = self.map_name_instance[key_orig]
                # This creates and adds the tree components
                self._add_buds(mapping_tree_structure=mapping_tree_structure,
                               mapping_lpy=mapping_lpy,
                               tree=tree,
                               parent_dict=branch_dict,
                               bud_sites=child_list)
                # This creates the SkeletonComponent for the branch
                self._add_skeleton(mapping_tree_structure=mapping_tree_structure,
                                   mapping_lpy=mapping_lpy,
                                   branch_dict=branch_dict,
                                   branch_lpy=branch_lpy,
                                   bud_sites=child_list)

        return tree, mapping_lpy, mapping_tree_structure

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
        export_dict = {"seed_value": int(self.extern_vars["seed_value"])}
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
