"""
Defines the abstract class BasicWood, class Wire and class Support.
"""

from abc import ABC, abstractmethod
from openalea.plantgl.all import *
import copy
import numpy as np
from openalea.plantgl.scenegraph.cspline import CSpline
import collections
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LocationState:
    """Location tracking for a wood object: start point, end point, and last tie location."""

    start: any = None  # Vector3
    end: any = None  # Vector3
    last_tie_location: any = None  # Vector3

    def __post_init__(self):
        """Initialize Vector3 points if not provided."""
        if self.start is None:
            self.start = Vector3(0, 0, 0)
        if self.end is None:
            self.end = Vector3(0, 0, 0)
        if self.last_tie_location is None:
            self.last_tie_location = Vector3(0, 0, 0)


@dataclass
class TyingState:
    """Tying and guiding state for a wood object."""

    has_tied: bool = False
    guide_points: list = None  # List of (x,y,z) tuples for spline control points
    guide_target: any = -1  # Wire object or -1 (no target)
    tie_axis: tuple = None  # Direction vector for the wire axis
    tie_updated: bool = False

    def __post_init__(self):
        """Initialize guide_points as empty list if not provided."""
        if self.guide_points is None:
            self.guide_points = []


@dataclass
class GrowthState:
    """Growth parameters for a wood object."""

    max_buds_segment: int = 5  # Total cumulative buds allowed across the entire branch segment (not per node)
    thickness: float = 0.1
    thickness_increment: float = 0.01
    growth_length: float = 1.0
    cylinder_length: float = 0.1
    max_length: float = 7.0


@dataclass
class InfoState:
    """Information/metadata for a wood object."""

    age: int = 0
    cut: bool = False
    prunable: bool = True
    order: int = 0
    num_branches: int = 0
    color: tuple = (0, 0, 0)  # RGB tuple for visualization
    material: int = 0
    branch_dict: any = None  # collections.deque

    def __post_init__(self):
        """Initialize branch_dict if not provided."""
        if self.branch_dict is None:
            self.branch_dict = collections.deque()


@dataclass
class BasicWoodConfig:
    """Configuration parameters for BasicWood initialization."""

    copy_from: any = None
    max_buds_segment: int = 5  # Total cumulative buds allowed across the entire branch segment (not per node)
    thickness: float = 0.1
    thickness_increment: float = 0.01
    growth_length: float = 1.0
    cylinder_length: float = 0.1  # Length of each individual cylinder
    max_length: float = 7.0
    tie_axis: tuple = None
    order: int = 0
    color: tuple = (0, 0, 0)
    material: int = 0
    prunable: bool = True
    name: str = None
    bud_spacing_age: int = 2  # Age interval for bud creation
    rng: np.random.Generator = None

    # Curve parameters for L-System growth guides
    curve_x_range: tuple = (-0.5, 0.5)  # X bounds for Bezier curve control points
    curve_y_range: tuple = (-0.5, 0.5)  # Y bounds for Bezier curve control points
    curve_z_range: tuple = (-1, 1)  # Z bounds for Bezier curve control points

    def __post_init__(self):
        """Validate geometric parameters for consistent growth behavior."""
        if self.growth_length is not None and self.cylinder_length is not None:
            if self.growth_length < self.cylinder_length:
                raise ValueError(
                    "BasicWoodConfig.growth_length must be >= cylinder_length "
                    f"(got {self.growth_length} < {self.cylinder_length})"
                )


class BasicWood(ABC):
    eps = 1e-6   # Default clip value for comparisons

    @staticmethod
    def clone(obj):
        try:
            return copy.deepcopy(obj)
        except copy.Error:
            raise copy.Error(f"Not able to copy {obj}") from None

    def __init__(self, config=None, copy_from=None, **kwargs):

        # Validate parameters
        if copy_from is None and config is None:
            raise ValueError("Either 'config' or 'copy_from' must be provided")

        # Handle config-based initialization
        if copy_from:
            self.__copy_constructor__(copy_from)
            return

        if not isinstance(config, BasicWoodConfig):
            raise ValueError("config must be provided when copy_from is None")

        self.rng = config.rng

        self.location = LocationState()
        # Tying variables
        self.tying = TyingState(tie_axis=config.tie_axis)
        self.current_tied = False
        # Information Variables
        self.info = InfoState(order=config.order, color=config.color, material=config.material, prunable=config.prunable)
        self.__length = 0
        # Growth Variables
        self.growth = GrowthState(
            max_buds_segment = config.max_buds_segment,
            thickness = config.thickness,
            thickness_increment = config.thickness_increment,
            growth_length = config.growth_length,
            cylinder_length = config.cylinder_length,
            max_length = config.max_length,
        )
        # Bud spacing for L-System rules
        self.bud_spacing_age = config.bud_spacing_age

        # Curve parameters for L-System growth guides
        self.curve_x_range = config.curve_x_range
        self.curve_y_range = config.curve_y_range
        self.curve_z_range = config.curve_z_range

    def __copy_constructor__(self, copy_from):
        update_dict = copy.deepcopy(copy_from.__dict__)
        for k, v in update_dict.items():
            setattr(self, k, v)
        # self.__dict__.update(update_dict)

    @abstractmethod
    def is_bud_break(self, num_buds_segment: int) -> bool:
        """This method defines if a bud will break or not -> returns true for yes, false for not. Input can be any variables"""
        pass
        # Example
        # prob_break = self.bud_break_prob_func(num_buds, self.num_buds_segment)
        # #Write dummy probability function
        # if prob_break > self.bud_break_prob:
        #   return True
        # return False

    @abstractmethod
    def pre_bud_rule(self, plant_segment, simulation_config) -> str:
        """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
        pass

    @abstractmethod
    def post_bud_rule(self, plant_segment, simulation_config) -> str:
        """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
        pass

    @abstractmethod
    def grow(self) -> None:
        """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
        pass

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        self.__length = min(length, self.growth.max_length)

    def grow_one(self):
        self.info.age += 1
        self.length += self.growth.growth_length
        self.grow()

    @abstractmethod
    def create_branch(self):
        """Returns how a new order branch when bud break happens will look like if a bud break happens"""
        pass
        # new_object = BasicWood.clone(self.branch_object)
        # return new_object
        # return BasicWood(self.num_buds_segment/2, self.bud_break_prob, self.thickness/2, self.thickness_increment/2, self.growth_length/2,\
        # self.max_length/2, self.tie_axis, self.bud_break_max_length/2, self.order+1, self.bud_break_prob_func)

    def update_guide(self, guide_target):
        """Compute and append guide control points for this wood object.

        Args:
            guide_target: Wire object (with .point attribute) or None/-1 (no-op).

        Notes:
            - If infeasible (tie point cannot be reached), silently returns.
            - Appends control points incrementally to self.tying.guide_points.
            - Uses self.location.start as base if not yet tied; self.location.last_tie_location otherwise.
        """
        self.tying.guide_target = guide_target
        if guide_target is None or guide_target == -1:
            return

        # Select base point: use last tie location if already tied, otherwise start
        base_point = self.location.last_tie_location if self.tying.has_tied else self.location.start

        # Compute control points and tie point in one call
        curve, tie_point = self.get_control_points(
            guide_target.point, base_point, self.location.end, self.tying.tie_axis
        )

        # Append only if feasible (tie_point is not None)
        if tie_point is not None and curve:
            self.tying.guide_points.extend(curve)
            # Note: last_tie_location updated at StartEach hook, not here

    def tie_lstring(self, lstring, index):
        """Insert a SetGuide(...) after position `index` in `lstring`.

        - Removes any immediate following tokens whose .name is in ('&','/','SetGuide').
        - Builds a CSpline from `self.tying.guide_points` and inserts the curve string and length.
        Returns (lstring, removed_count).
        """
        # Nothing to do if we don't have guide points
        if not self.tying.guide_points:
            return lstring, 0
        # Build spline and get curve representation (may raise)
        try:
            spline = CSpline(self.tying.guide_points)
            curve_repr = spline.curve(stride_factor=100)
        except Exception as exc:
            raise ValueError("Invalid spline from guide_points") from exc

        # Defensive check for 'nan' in the curve representation (preserve original check intent)
        if "nan" in str(curve_repr):
            raise ValueError("Curve is NaN", self.tying.guide_points)

        # Remove any immediate tokens after index that match the removal set
        removal_names = {"&", "/", "SetGuide"}
        insert_pos = index + 1
        removed_count = 0

        # Remove while the next token exists and matches
        while insert_pos < len(lstring) and getattr(lstring[insert_pos], "name", None) in removal_names:
            del lstring[insert_pos]
            removed_count += 1

        # Mark tied (if not already)
        if not self.tying.has_tied:
            self.tying.has_tied = True

        # Insert the new SetGuide token at the computed insert position
        lstring.insertAt(insert_pos, f"SetGuide({curve_repr}, {self.length})")

        return lstring, removed_count

    def tie_update(self):
        self.location.last_tie_location = copy.deepcopy(self.location.end)
        self.tying.tie_updated = True

    def deflection_at_x(self, d, x, L):
        """d is the max deflection, x is the current location we need deflection on and L is the total length"""
        return (d / 2) * (x**2) / (L**3) * (3 * L - x)

    # return d*(1 - np.cos(*np.pi*x/(2*L))) #Axial loading

    def get_control_points(self, target, start, current, tie_axis):
        """
        Compute control points for a 3D curve from branch segment to tie point on wire.

        Uses vector projection to determine feasibility and compute the tie point location,
        then generates a deflected curve using beam theory.

        Args:
            target: Wire point (x, y, z) - a point on the wire
            start: Branch segment start point (x, y, z)
            current: Branch segment end point (x, y, z)
            tie_axis: Unit direction vector of the wire (axis along which wire extends)

        Returns:
            tuple: (control_points, tie_point) where:
                - control_points: List of (x,y,z) tuples for curve fitting
                - tie_point: Computed tie location on wire, or None if infeasible

        Geometry:
            The branch, perpendicular offset to wire, and travel along wire form a right triangle:
            - Hypotenuse = branch_length (||current - start||)
            - One leg = perpendicular_distance (shortest distance from start to wire)
            - Other leg = parallel_travel (distance to travel along wire to reach it)
        """
        # Convert inputs to numpy arrays
        start_arr = np.array([start[0], start[1], start[2]], dtype=float)
        current_arr = np.array([current[0], current[1], current[2]], dtype=float)
        wire_point = np.array([target[0], target[1], target[2]], dtype=float)
        wire_axis = np.array(tie_axis, dtype=float)

        # Normalize wire axis to unit vector
        wire_axis_norm = np.linalg.norm(wire_axis)
        if wire_axis_norm < BasicWood.eps:
            return [], None
        wire_axis_unit = wire_axis / wire_axis_norm

        # Calculate branch segment length
        segment_vector = current_arr - start_arr
        branch_length = np.linalg.norm(segment_vector)
        if branch_length < BasicWood.eps:
            return [], None  # Degenerate segment

        # Vector from branch start to wire point
        v = wire_point - start_arr

        # Decompose v into components parallel and perpendicular to wire axis
        parallel_component, perpendicular_component = self._get_parallel_and_perpendicular_components(v, wire_axis_unit)
        perpendicular_distance = np.linalg.norm(perpendicular_component)

        # Feasibility check: branch must be long enough to reach the wire
        if perpendicular_distance > branch_length:
            return [], None

        # Calculate distance to travel along wire (Pythagorean theorem)
        # branch_length² = perpendicular_distance² + parallel_travel²
        parallel_travel_sq = branch_length**2 - perpendicular_distance**2
        parallel_travel = np.sqrt(max(0.0, parallel_travel_sq))  # Clamp to avoid floating-point negatives

        # Compute tie point on wire
        # Start from perpendicular projection of start onto wire, then move parallel_travel along wire
        start_projection_on_wire = start_arr + perpendicular_component
        direction_to_wire = np.sign(np.dot(wire_point, wire_axis_unit))
        tie_point = start_projection_on_wire + parallel_travel * wire_axis_unit * direction_to_wire

        # Generate control points along deflected curve using beam deflection formula
        control_points = self._generate_deflected_curve(start_arr, current_arr, tie_point)

        return control_points, tuple(tie_point)

    def _generate_deflected_curve(self, start, current, tie_point):
        control_points = []
        deflection_vector = np.array(tie_point) - np.array(current)
        branch_length = np.linalg.norm(np.array(current) - np.array(start))
        for step in np.arange(0.1, 1.1, 0.1):
            # Parametric position along branch segment [0.1, 0.2, ..., 1.0]
            t = step

            # Base position: linear interpolation from start to current
            base_position = start + t * (current - start)

            # Add beam deflection (cantilever formula)
            deflection = self.deflection_at_x(deflection_vector, t * branch_length, branch_length)

            # Combine base position and deflection
            point = tuple(base_position + deflection)
            control_points.append(point)
        return control_points

    def _get_parallel_and_perpendicular_components(self, vec_a, vec_b):
        # Project vec_a onto vec_b to get parallel and perpendicular components
        vec_b_unit = vec_b / np.linalg.norm(vec_b)
        parallel_component = np.dot(vec_a, vec_b_unit) * vec_b_unit
        perpendicular_component = vec_a - parallel_component
        return parallel_component, perpendicular_component


class TreeBranch(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    count = 0    # Class variable for instance counting

    def __init__(
        self,
        config=None,
        copy_from=None,
        prototype_dict: dict=None,
        name: str = None,
        contour_params: tuple = (1, 0.2, 30),
    ):
        # Validate parameters
        if copy_from is None and config is None:
            raise ValueError("Either 'config' or 'copy_from' must be provided")

        # Call BasicWood constructor
        super().__init__(config, copy_from)

        # Handle copy construction vs new instance
        if copy_from:
            # BasicWood already handled the copy, just set additional attributes
            pass
        else:
            if prototype_dict == None:
                self.prototype_dict = {}
            else:
                self.prototype_dict = prototype_dict

        # Set name with automatic numbering
        if not name:
            self.name = f"{self.__class__.__name__}_{self.__class__.count}"
        self.__class__.count += 1

        # Set up contour (subclasses can override contour_params)
        radius, noise_factor, num_points = contour_params
        self.contour = None  # create_noisy_branch_contour(radius, noise_factor, num_points)

        # Initialize common attributes
        self.num_buds = 0

        # Initialize subclass-specific attributes
        self._init_subclass_attributes()

    def _init_subclass_attributes(self):
        """Hook for subclasses to initialize their specific attributes"""
        pass

    def grow(self):
        """Default empty implementation - subclasses can override if needed"""
        pass


@dataclass
class Wire:
    # All wires are horizontal, tying axis depends on wood definition
    id: int
    point: Tuple[float, float, float]
    num_branch: int = 0

    def add_branch(self):
        self.num_branch += 1


class Support:
    """All the details needed to figure out how the support is structured in the environment, it is a collection of wires"""

    def __init__(
        self,
        points: list,
        num_wires: int,
        spacing_wires: int,
        trunk_wire_pt: tuple[float, float, float],
    ):

        self.num_wires = num_wires
        self.spacing_wires = spacing_wires
        self.branch_supports = self.make_support(points)  # Dictionary id:points
        self.trunk_wire = None
        if trunk_wire_pt:
            self.trunk_wire = Wire(id=-1, point=trunk_wire_pt)
            points.append(trunk_wire_pt)

        self.attractor_grid = Point3Grid((1, 1, 1), list(points))

    def make_support(self, points):
        supports = {}
        for id, pt in enumerate(points):
            supports[id] = Wire(id, pt)
        return supports
