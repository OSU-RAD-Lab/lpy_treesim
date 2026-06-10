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


def angle_between(angle, min_angle, max_angle):
    """
    Check if an angle falls within a specified range after 90-degree offset.

    Applies a 90-degree offset to the input angle and checks if the result
    falls within the specified range. This is used for determining acceptable
    tropism angles in the pruning strategy.

    Args:
        angle: Input angle in degrees
        min_angle: Minimum angle of the acceptable range (after offset)
        max_angle: Maximum angle of the acceptable range (after offset)

    Returns:
        bool: True if the offset angle is within the range, False otherwise
    """
    offset_angle = angle + 90
    return min_angle <= offset_angle <= max_angle


def generate_random_offset(radius: float, rng: np.random.Generator) -> float:
    """
    Generate a random offset value within a specified radius range.

    Creates a random float value between -radius and +radius, useful for
    adding noise or variation to geometric shapes and curves.

    Args:
        radius: Maximum absolute value for the random offset

    Returns:
        float: Random value between -radius and +radius
    """
    return rng.uniform(-radius, radius)


def generate_noisy_branch_curve(radius: float, rng: np.random.Generator, num_control_points: int = 20):
    """
    Generate a NURBS curve representing a noisy branch shape.

    Creates a 3D NURBS curve with noise applied to create a natural-looking
    branch shape. The curve starts at the origin and extends along the z-axis,
    with x and y coordinates perturbed by noise that scales with distance.

    Args:
        radius: Base radius for noise generation
        num_control_points: Number of control points for the NURBS curve

    Returns:
        NurbsCurve: PlantGL NURBS curve object representing the noisy branch
    """
    # Create control points with progressive noise
    control_points = [(0, 0, 0, 1), (0, 0, 1 / float(num_control_points - 1), 1)]

    for point_index in range(2, num_control_points):
        t = point_index / float(num_control_points - 1)
        noise_scale = radius * 2  # amplitude scaling factor

        x_noise = generate_random_offset(radius=noise_scale, rng=rng)
        y_noise = generate_random_offset(radius=noise_scale, rng=rng)

        control_points.append((x_noise, y_noise, t, 1))

    return NurbsCurve(control_points, degree=min(num_control_points - 1, 3), stride=num_control_points * 100)


def create_noisy_branch_contour(
    radius,
    noise_factor,
    rng: np.random.Generator,
    num_points=100,
):
    """
    Create a noisy 2D contour for branch cross-sections.

    Generates a circular contour with added noise to create natural-looking
    branch cross-section shapes. The contour is closed and can be used for
    extruding 3D branch geometry.

    Args:
        radius: Base radius of the circular contour
        noise_factor: Scale factor for the noise added to the contour
        num_points: Number of points in the contour (higher = smoother)
        seed_value: Random seed for reproducible results

    Returns:
        Polyline2D: PlantGL 2D polyline representing the noisy contour
    """

    # Generate angles around the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    contour_points = []

    for angle in angles:
        # Calculate base circle coordinates
        x_base = radius * np.cos(angle)
        y_base = radius * np.sin(angle)

        # Add noise to create irregular shape
        x_noise = rng.uniform(-noise_factor, noise_factor)
        y_noise = rng.uniform(-noise_factor, noise_factor)

        x_noisy = x_base + x_noise
        y_noisy = y_base + y_noise

        contour_points.append((x_noisy, y_noisy))

    # Close the contour by repeating the first point
    contour_points.append(contour_points[0])

    # Create PlantGL geometry
    point_array = Point2Array(contour_points)
    return Polyline2D(point_array)


def create_bezier_curve(
    num_control_points=6, x_range=(-2, 2), y_range=(-2, 2), z_range=(0, 10), rng: np.random.Generator = None
) -> BezierCurve:
    """
    Create a randomized 3D Bezier curve for growth guidance.

    Generates a Bezier curve with randomly positioned control points within
    specified ranges. The curve progresses along the z-axis with control points
    distributed evenly in the z-direction but randomly in x and y.

    Args:
        num_control_points: Number of control points for the Bezier curve
        x_range: Tuple (min_x, max_x) defining the x-coordinate range
        y_range: Tuple (min_y, max_y) defining the y-coordinate range
        z_range: Tuple (min_z, max_z) defining the z-coordinate range
        seed_value: Random seed for reproducible curve generation

    Returns:
        BezierCurve: PlantGL Bezier curve object for growth guidance
    """

    # Generate control points with progressive z-coordinates
    z_values = np.linspace(z_range[0], z_range[1], num_control_points)
    control_points = []

    for z_value in z_values:
        x_coord = rng.uniform(x_range[0], x_range[1])
        y_coord = rng.uniform(y_range[0], y_range[1])
        control_points.append(Vector4(x_coord, y_coord, z_value, 1))

    # Create PlantGL Bezier curve
    control_point_array = Point4Array(control_points)
    return BezierCurve(control_point_array)


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
