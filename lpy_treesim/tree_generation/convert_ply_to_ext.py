import os

import argparse
import glob
from lpy_treesim.tree_generation.lpy_scene_to_mesh import convert_ply_to_ext

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dataset/")
    parser.add_argument("--input_dir", type=str, default="tree_models/UFO_tie_prune_label.lpy")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--ext", type=str, default="obj")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in glob.glob(os.path.join(args.input_dir, "*.ply")):
        output_path = os.path.join(args.output_dir, os.path.basename(file).replace(".ply", "." + args.ext))
        if args.verbose:
            print("INFO: Converting file: ", file, "and saving to: ", output_path)
        convert_ply_to_ext(file, output_path)
