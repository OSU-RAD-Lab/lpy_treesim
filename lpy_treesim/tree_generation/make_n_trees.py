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
from lpy_treesim.tree_generation.tree_to_usd import check_texture # create_mesh_usd
from lpy_treesim.textures.generate_texture import make_texture_set # make_uv_texture
#from lpy_treesim.tree_generation.tree_structure import calculate_skeleton_junctions

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save multiple L-Py trees.")
    parser.add_argument("--num-trees", type=int, default=1, help="Number of trees to generate")
    parser.add_argument("--stage-dir", type=Path, default=Path("."), help="Directory for top of Stage USD files")
    parser.add_argument("--output-dir", type=Path, default=Path("./data"), help="Directory for regular mesh outputs")
    parser.add_argument("--tree-name", type=str, default="envy", help="Tree family to generate (UFO/Envy/etc.)")
    parser.add_argument("--texture-name", type=str, default="apple", help="Use/make all textures with this name")
    parser.add_argument("--verbose", action="store_true", help="Print progress details")
    parser.add_argument("--interactive", action="store_true", help="Show tree growing")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Optional deterministic seed for dataset generation")
    parser.add_argument("--namespace", type=str, default="lpy", help="Prefix namespace for output filenames")
    parser.add_argument("--ply", action="store_true", help="Write out ply file format")
    parser.add_argument("--obj", action="store_false", help="Write out obj file format")
    parser.add_argument("--meta-data", action="store_true", help="Write out meta data")
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
    os.makedirs(args.stage_dir, exist_ok=True)

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
        lsb = TreeBuilder(tree_name=args.tree_name, 
                          seed_value=int(tree_seed),
                          args=args)

        if args.verbose:
            print(f"INFO: Generating {args.tree_name} tree #{index:03d}")
        logging.info(f"Generating {args.tree_name} tree #{index:03d} with seed {tree_seed}")

        # Generates the l-string that everything is built off of, then converts it to the "scene"
        #   Also sets one color for each spur/branch/trunk instance (stored in branch_hierarchy)
        usd_path, mesh_path = lsb.generate_tree(naming=naming, 
                                     index=index, 
                                     radii=radii, 
                                     name_radii=name_radii,
                                     stage_context=stage_context)

        logger.info(f"Wrote mesh to {usd_path}")
        logger.info(f"Wrote ply/obj to {mesh_path}")

        # Create indicators, from marked locations, on the tree after 
        # the tree has been generated
    
        # Create path for marked locations file and load generated tree mesh
        # Added this code. This gets the base name and appends the file to _vc.obj 
        # so trimesh will find the file
        mesh_name = naming.mesh_filename(index, file_type="") + "_vc.obj"
        mod_path = str(args.output_dir / mesh_name)
        mesh_existing = trimesh.load(mod_path)
        # all_meshes = [mesh_existing]
        try:
            df = pd.read_csv("lpy_treesim/tie_prune/pruning_algo/ltr_marked_locations.csv")
            
            # Loop through each iteration and saves in the CSV
            for iter_val in df['iteration'].unique():
                # Filter the dataframe to only include spheres for this specific iteration
                iter_df = df[df['iteration'] == iter_val]
                
                # Create a fresh list with a clean tree for this specific year
                iter_meshes = [mesh_existing.copy()]
                
                for x, y, z, rad in zip(iter_df['marked_x'], iter_df['marked_y'], iter_df['marked_z'], iter_df['radius']):
                    marker = trimesh.creation.icosphere(subdivisions=2, radius=rad)
                    marker.visual.face_colors = [255, 165, 0, 200]
                    marker.visual = marker.visual.to_texture()
                    marker.visual.material.alphaMode = "BLEND"
                    marker.apply_translation((x, y, z))
                    
                    # Append the sphere to this year's fresh list
                    iter_meshes.append(marker)
                
                # Combine the tree and spheres for this specific iteration
                combined_mesh = trimesh.util.concatenate(iter_meshes)
                
                # Export as a separate file (e.g., marked_location_iter_29.obj)
                out_name = f"{str(args.output_dir)}/marked_location_iter_{int(iter_val)}.obj"
                combined_mesh.export(out_name)
                print(f"Exported year file: {out_name}")
                
        except Exception as e:
            print(f"Skipping sphere generation. Error: {e}")


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

  
    logger.info("Tree generation complete.")
    return


if __name__ == "__main__":
    main()
