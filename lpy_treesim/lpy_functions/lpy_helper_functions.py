"""
Helper utilities for L-Py tree simulation system.

This module provides utility functions for procedural tree generation and simulation
in the L-Py framework. It includes functions for:

- L-System string manipulation (cutting, pruning operations)
- Geometric shape generation (contours, curves, noise patterns)
- Tree training and optimization utilities
- PlantGL integration for 3D visualization

The functions in this module are used by various tree architecture implementations
(UFO, Envy, etc.) to perform common operations like branch pruning, wire attachment
optimization, and geometric shape generation for realistic tree modeling.
"""

from openalea.plantgl.all import (
    NurbsCurve,
    Vector3,
    Vector4,
    Point4Array,
    Point2Array,
    Point3Array,
    Polyline2D,
    BezierCurve,
    BezierCurve2D,
)
from openalea.lpy import Lsystem, newmodule
import numpy as np
from typing import Callable, Dict, Iterable
import importlib
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import TreeBranch
from lpy_treesim.tie_prune.tie_prune_simulation_base import TreeSimulationBase

def should_bud(plant_segment, simulation_config):
    """Determine if a plant segment should produce a bud"""
    return np.isclose(plant_segment.info.age % plant_segment.bud_spacing_age, 0, atol=simulation_config.tolerance)


def start_each_common(lstring,
                      branch_hierarchy: Dict[str, Iterable],
                      tree_sim: TreeSimulationBase,
                      main_trunk: TreeBranch,
):
    """Shared pre-iteration tying preparation logic.
    @param lstring - the actual lstring being generated
    @param branch_hierarchy - the current branch hierarchy as a dictionary
    @param trellis_support - the trellis support (which has the attractor grids)
    @param main_trunk - the trunk, which inherits from TreeBranch """

    # del lstring  # unused in shared logic; kept for L-Py parity
    # If we haven't added the attractor for the main trunk, do so
    if not main_trunk.tying.wire_attach and len(tree_sim.trunk_attractor) > 0:
        main_trunk.tying.wire_attach = tree_sim.trunk_attractor[0]

    """
    # First level branches
    for branch in branch_hierarchy[main_trunk.name]:
        if not branch.tying.tie_needs_updating:
            branch.set_tie_update()
    """
    return lstring

def end_each_common(lstring,
                    branch_hierarchy: Dict[str, Iterable],
                    tree_sim: TreeSimulationBase,
                    tying_interval_iterations: int,
                    pruning_interval_iterations: int,
                    simulation_config,
                    main_trunk,
                    get_iteration_number: Callable[[], int],
):
    """Shared post-iteration tying and pruning orchestration."""
    current_iteration = get_iteration_number() + 1

    if current_iteration % tying_interval_iterations == 0:
        if tree_sim.trunk_attractor:
            main_trunk.update_guide()

        branches = branch_hierarchy[main_trunk.name]
        # Estimate of cost to tie branches to open wire attachments
        energy_matrix, open_branches = tree_sim.get_energy_matrix(branches)

        # Actually tie some of the branches to the wires
        tree_sim.decide_guide(energy_matrix, open_branches)

        # Update guide curve
        for branch in branches:
            branch.update_guide()

        # This does the actual tying
        while tree_sim.tie(lstring):
            pass

    if current_iteration % pruning_interval_iterations == 0:
        while tree_sim.prune(lstring, branch_hierarchy):
            pass

    return lstring


def resolve_attr(path: str):
    """Import a fully qualified attribute path."""
    pkg_path, attr_name = path.rsplit(".", 1)
    pkg = importlib.import_module(pkg_path)
    return getattr(pkg, attr_name)
