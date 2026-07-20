#! /usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import secrets
import os as os
import logging
import trimesh
import pandas as pd

from lpy_treesim.tree_generation.tree_builder_lpy import TreeBuilder
from lpy_treesim.tree_generation.file_naming_config import FileNamingConfig
from lpy_treesim.tree_generation.tree_to_usd import create_mesh_usd, check_texture
from lpy_treesim.textures.generate_texture import make_texture_set, make_uv_texture
from lpy_treesim.tree_generation.lpy_scene_to_mesh import plant_gl_scene_to_vertices_and_faces, stitch_cylinders, write_mesh
from lpy_treesim.tree_generation.tree_structure import calculate_skeleton_junctions

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save multiple L-Py trees.")
    parser.add_argument("--num-trees", type=int, default=1, help="Number of trees to generate")
    parser.add_argument("--stage-dir", type=Path, default=None, help="Directory for top of Stage USD files")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/"), help="Directory for regular mesh outputs")
    parser.add_argument("--tree-name", type=str, default="ufo", help="Tree family to generate (UFO/Envy/etc.)")
    parser.add_argument("--texture-name", type=str, default="apple", help="Use/make all textures with this name")
    parser.add_argument("--verbose", action="store_true", help="Print progress details")
    parser.add_argument("--interactive", action="store_true", help="Show tree growing")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Optional deterministic seed for dataset generation")
    parser.add_argument("--namespace", type=str, default="lpy", help="Prefix namespace for output filenames")
    parser.add_argument("--ply", action="store_false", help="Write out ply file format")
    parser.add_argument("--obj", action="store_false", help="Write out obj file format")
    parser.add_argument("--meta-data", action="store_false", help="Write out meta data")
    parser.add_argument("--usda", action="store_true", help="Write out universal scene descriptor format")
    parser.add_argument("--make-textures", action="store_true", help="Create a new set of textures")
    args = parser.parse_args()

    if args.num_trees > (FileNamingConfig.MAX_TREES + 1) or args.num_trees < 1:
        raise ValueError(f"num_trees={args.num_trees} is not in the range [1, {FileNamingConfig.MAX_TREES + 1}].")
    if args.dataset_seed is None:
        args.dataset_seed = secrets.randbits(32)
    return args


def main():
    logger.info("Starting tree generation...")
    args = _parse_args()

    naming = FileNamingConfig(namespace=args.namespace, tree_type=args.tree_name)
    # ensure_output_dir(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    stage_context = []
    if args.stage_dir is not None:
        from pxr import Ar, Usd
        # 1. Define your search paths
        loc_name = str(args.stage_dir)
        os.chdir(loc_name)
        print(f"Changing to {loc_name}")
        search_paths = [loc_name]

        # 2. Create a context with these paths
        stage_context = Ar.DefaultResolverContext(search_paths)

        check_texture(stage_context)

        if args.make_textures:
            radii, name_radii = make_texture_set(args.tree_name, loc_name + "/textures")
        else:
            radii = [1]
            name_radii = ["pine_bark_vmbibe2g_2k"]

    # Generate trees - use unique seeds
    tree_rng: np.random.Generator = np.random.default_rng(seed=args.dataset_seed)
    for index in range(args.num_trees):
        # Seed
        tree_seed = tree_rng.integers(low=0, high=1_000_000)

        # Initialize the class
        lsb = TreeBuilder(tree_name=args.tree_name, seed_value=int(tree_seed), interactive=args.interactive)

        if args.verbose:
            print(f"INFO: Generating {args.tree_name} tree #{index:03d}")
        logging.info(f"Generating {args.tree_name} tree #{index:03d} with seed {tree_seed}")

        # Generates the l-string that everything is built off of, then converts it to the "scene"
        #   Also sets one color for each spur/branch/trunk instance (stored in branch_hierarchy)
        lstring, scene = lsb.generate_tree()

        # Converts the scene to our tree structure.
        #   Mapping maps the unique ids from the lstring into our tree structure
        #   This ensures the branches etc are numbered sequentially
        tree, mapping_lpy, mapping_tree_structure = lsb.create_tree_structure()

        # Adds to each tree component the mesh cylinders created by lpy
        bud_sites = plant_gl_scene_to_vertices_and_faces(scene,
                                                         mapping_tree_structure=mapping_tree_structure,
                                                         color_mapping=lsb.color_manager)

        # Now stitch together all of the mesh components into tubes instead of discrete cylinders
        # Also adds colors and texture coordinates
        color_to_part, keys_to_remove = stitch_cylinders(tree=tree)
        # Some newly created branch parts do not have any meshes associated with them
        #for key in keys_to_remove:
        #    tree.remove_key(key)

        # Now that the cylinders/skeleton have been processed, build the junctions
        #calculate_skeleton_junctions(tree=tree)

        # Write out mesh file formats
        if args.ply or args.obj:
            mesh_path = args.output_dir / naming.mesh_filename(index, file_type="")
            uv_name = str(mesh_path) + "_uv.png"
            make_uv_texture(uv_name)
            write_mesh(tree=tree, fname=mesh_path, bud_sites=bud_sites, image_name=uv_name)

        if stage_context is not [] and args.usda:
            # Where the usd files are stored
            usd_path = args.stage_dir / naming.usd_filename(index)
            uv_name = str(args.stage_dir ) + "/textures/mesh_uv.png"
            make_uv_texture(uv_name)
            for b_use_uv in [True, False]:
                create_mesh_usd(stage_context, 
                                world_path=str(args.stage_dir), 
                                in_tree_name=naming._prefix(index), 
                                tree=tree, 
                                radii=radii, name_radii=name_radii,
                                b_use_uv=b_use_uv)
            logger.info(f"Wrote mesh to {usd_path}")


        # Create indicators, from marked locations, on the tree after 
        # the tree has been generated
    
        # Create path for marked locations file and load generated tree mesh
        # Added this code. This gets the base name and appends the file to _vc.obj 
        # so trimesh will find the file
        mesh_name = naming.mesh_filename(index, file_type="") + "_vc.obj"
        mod_path = str(args.output_dir / mesh_name)
        mesh_existing = trimesh.load(mod_path)
        all_meshes = [mesh_existing]

        try:
            df = pd.read_csv(Path(__file__).parents[1].resolve() / "tie_prune" / "marked_locations.csv")
            #mark locations, with transparent sphere, specified in prune_at_end.py
            for x, y, z in zip(df['marked_x'], df['marked_y'], df['marked_z']):
                marker = trimesh.creation.icosphere(subdivisions=2, radius=df['radius'].iloc[0])
                marker.visual.face_colors = [0, 255, 0, 35]
                marker.visual = marker.visual.to_texture()
                marker.visual.material.alphaMode = "BLEND"
                marker.apply_translation((x, y, z))
                all_meshes.append(marker)
                
        except:
            print(Path(__file__).parents[1].resolve() / "tie_prune" / "marked_locations.csv" "not found")
        #Added this code just playing around with it 
        # 1. Draw the Nodes (The Buds)
        try:
            df_nodes = pd.read_csv("data/knn_nodes.csv")
            for _, row in df_nodes.iterrows():
                # Make a tiny sphere for the node
                node = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
                
                # Color it green if it's the reference, blue if it's a neighbor
                if row['type'] == 'ref':
                    node.visual.face_colors = [0, 255, 0, 255] 
                else:
                    node.visual.face_colors = [0, 0, 255, 255]
                    
                node.apply_translation((row['x'], row['y'], row['z']))
                all_meshes.append(node)
        except FileNotFoundError:
            print("No knn_nodes.csv found. Skipping KD-Tree node visualization.")
        
        try:
            df_edges = pd.read_csv("data/knn_edges.csv")
            for _, row in df_edges.iterrows():
                start_pt = [row['x1'], row['y1'], row['z1']]
                end_pt = [row['x2'], row['y2'], row['z2']]
                
                # Create a thin 3D cylinder acting as a line connecting the two points
                line = trimesh.creation.cylinder(radius=0.005, segment=[start_pt, end_pt])
                line.visual.face_colors = [255, 0, 0, 150] # Semi-transparent red
                all_meshes.append(line)
        except FileNotFoundError:
            print("No knn_edges.csv found. Skipping KD-Tree edge visualization.")
        # End of my added additions 

        #combine meshes and save file
        combined_mesh = trimesh.util.concatenate(all_meshes)
        combined_mesh.export(str(args.output_dir)+"/marked_location.obj")


        if args.meta_data:
            import json
            metadata_path = args.output_dir / naming.metadata_filename(index)
            meta_data = lsb.get_metadata()
            meta_data["tree"] = tree.create_dict()
            # meta_data["tree_mapping"] = mapping
            meta_data["color_mapping"] = color_to_part
            with open(metadata_path, "w") as f:
                json.dump(meta_data, f, indent=4)
            logger.info(f"Wrote meta data to {metadata_path}")

        del scene
        del lstring
        del lsb
    logger.info("Tree generation complete.")
    return


if __name__ == "__main__":
    main()
