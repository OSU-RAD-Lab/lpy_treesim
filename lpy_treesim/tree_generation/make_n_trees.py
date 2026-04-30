#! /usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import secrets
import os as os



import logging
import lpy_treesim.utils.logging_conf
from lpy_treesim.tree_generation.tree_builder import TreeBuilder
from lpy_treesim.tree_generation.tree_name_conf import TreeNamingConfig
from lpy_treesim.tree_generation.convert_ply_to_usd import create_mesh_usd, check_texture
import lpy_mesh_utils as lmu

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save multiple L-Py trees.")
    parser.add_argument("--num-trees", type=int, default=1, help="Number of trees to generate")
    parser.add_argument("--stage-dir", type=Path, default=Path("/home/cindy/isaacsim/World"), help="Directory for top of Stage USD files")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/cindy/VSCode/data/lpy_trees/"), help="Directory for regular mesh outputs")
    parser.add_argument("--tree-name", type=str, default="envy", help="Tree family to generate (UFO/Envy/etc.)")
    parser.add_argument("--verbose", action="store_true", help="Print progress details")
    parser.add_argument(
        "--dataset-seed", type=int, default=None, help="Optional deterministic seed for dataset generation"
    )
    parser.add_argument("--namespace", type=str, default="lpy", help="Prefix namespace for output filenames")
    parser.add_argument("--semantic-label", action="store_true", help="Enable semantic labeling")
    parser.add_argument("--instance-label", action="store_true", help="Enable instance labeling")
    parser.add_argument("--per-cylinder-label", action="store_true", help="Enable per-cylinder labeling")
    args = parser.parse_args()
    if args.num_trees > (TreeNamingConfig.MAX_TREES + 1) or args.num_trees < 1:
        raise ValueError(f"num_trees={args.num_trees} is not in the range [1, {TreeNamingConfig.MAX_TREES + 1}].")
    if args.dataset_seed is None:
        args.dataset_seed = secrets.randbits(32)
    return args


def main():
    logger.info("Starting tree generation...")
    args = _parse_args()

    naming = TreeNamingConfig(namespace=args.namespace, tree_type=args.tree_name)
    # ensure_output_dir(args.output_dir)

    stage_context = []
    if args.stage_dir is not None:
        from pxr import Ar, Usd
        # 1. Define your search paths
        loc_name = str(args.stage_dir)
        os.chdir(loc_name)
        print(f"Changing to {loc_name}")
        search_paths = [str(args.stage_dir)]
        search_paths = [loc_name]

        # 2. Create a context with these paths
        stage_context = Ar.DefaultResolverContext(search_paths)

        check_texture(stage_context)

    # Seeds
    # 292206 - no branches
    # 458657 - two branches, one empty
    # Generate trees
    tree_rng: np.random.Generator = np.random.default_rng(seed=args.dataset_seed)
    for index in range(args.num_trees):
        tree_seed = tree_rng.integers(low=0, high=1_000_000)
        lsb = TreeBuilder(
            tree_name=args.tree_name,
            seed_value=tree_seed,
            semantic_label=args.semantic_label,
            instance_label=args.instance_label,
            per_cylinder_label=True
            #per_cylinder_label=args.per_cylinder_label,
        )

        if args.verbose:
            print(f"INFO: Generating {args.tree_name} tree #{index:03d}")
        logging.info(f"Generating {args.tree_name} tree #{index:03d} with seed {tree_seed}")

        lstring, scene = lsb.generate_tree(b_interactive=False)
        # PLY
        mesh_path = args.output_dir / naming.mesh_filename(index)
        metadata_path = args.output_dir / naming.metadata_filename(index)
        usd_path = args.stage_dir / naming.usd_filename(index)

        mesh_components = lmu.plant_gl_scene_to_vertices_and_faces(scene)
        meta_data = lsb.get_metadata(mesh_components)
        lsb.export_metadata(mesh_components, metadata_path=str(metadata_path))
        tree, mapping = lsb.create_tree_structure()
        meta_data["tree"] = tree
        meta_data["tree_mapping"] = mapping

        mesh_components_ordered = lmu.stitch_cylinders(mesh_components, meta_data)
        if stage_context is not []:
            # Where the usd files are stored
            create_mesh_usd(stage_context, naming._prefix(index), usd_path, mesh_components_ordered, meta_data)

        # Write the metadata/mesh to the output directory
        lmu.write(str(mesh_path), mesh_components)

        # Metadata

        logger.info(f"Wrote mesh to {mesh_path} and metadata to {metadata_path}")
        del scene
        del lstring
        del lsb
    logger.info("Tree generation complete.")
    return


if __name__ == "__main__":
    main()
