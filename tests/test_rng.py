#! /usr/bin/env python3
import numpy as np
import os
import plyfile
import pytest

from lpy_treesim.tree_generation.tree_builder import TreeBuilder
from lpy_treesim.tree_generation.tree_builder import TreeNamingConfig
import lpy_treesim.tree_generation.lpy_scene_to_mesh as lmu

import logging
logger = logging.getLogger(__name__)

# def test_deterministic_generation():
#     import hashlib
#     seed = 12345
#     tree_name = "UFO"

#     lsb1 = TreeBuilder(tree_name=tree_name, seed_value=seed)
#     lstring1, scene1 = lsb1.generate_tree()
#     lsb2 = TreeBuilder(tree_name=tree_name, seed_value=seed)
#     lstring2, scene2 = lsb2.generate_tree()
#     print(lstring1.to_string())
#     assert hashlib.sha256(str(lstring1).encode('utf-8')).digest() == hashlib.sha256(str(lstring2).encode('utf-8')).digest(), "L-strings do not match for the same seed"
#     # assert scene1 == scene2, "Scenes do not match for the same seed"
#     return

# def test_different_seeds():
#     seed1 = 12345
#     seed2 = 54321
#     tree_name = "UFO"

#     lsb1 = TreeBuilder(tree_name=tree_name, seed_value=seed1)
#     lstring1, scene1 = lsb1.generate_tree()
#     lsb2 = TreeBuilder(tree_name=tree_name, seed_value=seed2)
#     lstring2, scene2 = lsb2.generate_tree()

#     assert lstring1 != lstring2, "L-strings should differ for different seeds"
#     assert scene1 != scene2, "Scenes should differ for different seeds"
#     return

def test_deterministic_ply_generation():
    seed = 12345
    tree_name = "UFO"
    output_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(output_dir, exist_ok=True)
    naming = TreeNamingConfig(namespace="lpy", tree_type=tree_name)

    tb1 = TreeBuilder(tree_name=tree_name, seed_value=seed)
    lstring1, scene1 = tb1.generate_tree()
    # PLY
    mesh_path = os.path.join(output_dir, naming.mesh_filename(index=0))
    lmu.write(str(mesh_path), scene1)

    tb2 = TreeBuilder(tree_name=tree_name, seed_value=seed)
    lstring2, scene2 = tb2.generate_tree()
    # PLY
    mesh_path2 = os.path.join(output_dir, naming.mesh_filename(index=1))
    lmu.write(str(mesh_path2), scene2)

    ply_data1 = plyfile.PlyData.read(str(mesh_path))
    ply_data2 = plyfile.PlyData.read(str(mesh_path2))

    (x1,y1,z1, r1, g1, b1) = (
        ply_data1['vertex']['x'],
        ply_data1['vertex']['y'],
        ply_data1['vertex']['z'],
        ply_data1['vertex']['red'],
        ply_data1['vertex']['green'],
        ply_data1['vertex']['blue'],
    )

    (x2,y2,z2, r2, g2, b2) = (
        ply_data2['vertex']['x'],
        ply_data2['vertex']['y'],
        ply_data2['vertex']['z'],
        ply_data2['vertex']['red'],
        ply_data2['vertex']['green'],
        ply_data2['vertex']['blue'],
    )
    logger.debug("----")
    logger.debug(len(x1))
    logger.debug(len(y1))
    logger.debug(len(z1))
    logger.debug(len(r1))
    logger.debug(len(g1))
    logger.debug(len(b1))
    logger.debug(len(x2))
    logger.debug(len(y2))
    logger.debug(len(z2))
    logger.debug(len(r2))
    logger.debug(len(g2))
    logger.debug(len(b2))
    assert np.all(x1 == x2), "X coordinates do not match for the same seed"
    return
