"""
Defines the abstract class BasicWood along with helper classes LocationState, GrowthState, InfoState and TyingState
"""

from abc import ABC, abstractmethod
from openalea.plantgl.all import *
import copy
import numpy as np
from openalea.plantgl.scenegraph.cspline import CSpline
import collections
from dataclasses import dataclass
from enum import Enum

from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tie_prune.tie_prune_configuration import SimulationConfig
import logging


@dataclass
class LocationState:
    """Location tracking for a wood object: start point & direction, end point and direction
    These values are filled in from the string during the interpretation stage"""

    start: any = None      # Vector3
    start_dir: any = None  # Vector3
    end: any = None        # Vector3
    end_dir: any = None    # Vector3

    def __post_init__(self):
        """Initialize Vector3 points if not provided."""
        if self.start is None:
            self.start = Vector3(0, 0, 0)
        if self.start_dir is None:
            self.start_dir = Vector3(1.0, 0, 0)
        if self.end is None:
            self.end = Vector3(0, 0, 0)
        if self.end_dir is None:
            self.end_dir = Vector3(0, 0, 0)


@dataclass
class GrowthState:
    """Growth parameters for a wood object. These will be set when constructed but with
       some noise, depending on the overall vigor level assigned to the branch"""

    # Set these
    vigour_level: float = 0.5  # Between 0 and 1
    length: float = 0.0
    config: SimulationConfig = None  # Random number generator and number of iteration steps are in here

    # These will be set to values based on vigour and default
    mean_length_growth_per_year: list[float] = None
    thickness: float = 0.001
    mean_thickness_growth_per_iteration: float = 0.1
    target_thickness_ratio = 0.025 / 1.0   # Ideal

    # For tracking growth; year will increment when a year completes
    year: int = 0
    current_iteration: int = 0

    def __post_init__(self):
        if self.mean_length_growth_per_year is None:
            #  Really should never get here... but between 0.5 and 1.5 meters
            self.mean_length_growth_per_year = [(1.0 - self.vigour_level) * 0.5 + self.vigour_level * 1.5]
        self.set_thickness_target()

    def set_thickness_target(self):
        """Calculate the target thickness based on the vigor level and expected growth for the year and current length
           Only call on year boundaries """
        estimated_length = self.length + self.length_growth_per_year()
        low_ratio = 0.01 / 1.0
        ideal_ratio = 0.025 / 1.0
        high_ratio = 3.5 / 1.0
        if self.vigour_level < 0.5:
            t = self.vigour_level * 2.0
            ratio = (1 - t) * low_ratio + t * ideal_ratio
        else:
            t = (self.vigour_level - 0.5) * 2.0
            ratio = (1 - t) * ideal_ratio + t * high_ratio
        if self.length > 0.0:
            cur_ratio = self.thickness / self.length_growth_per_year()
        else:
            cur_ratio = 0.0
        self.mean_thickness_growth_per_iteration = cur_ratio * (estimated_length) - self.thickness

    def length_growth_per_year(self):
        if self.year > len(self.mean_length_growth_per_year):
            return self.mean_length_growth_per_year[-1]
        return self.mean_length_growth_per_year[self.year]

    def mean_length_growth_per_iteration(self):
        return self.length_growth_per_year() / self.config.num_iter_per_year

    def get_thickness_increment(self):
        thick_incr = self.config.lpy_rng.normal(loc=self.mean_thickness_growth_per_iteration, scale=0.1 * self.mean_thickness_growth_per_iteration)
        return thick_incr

    def get_length_increment(self):
        mean_length_growth = self.mean_length_growth_per_iteration()
        length_incr = self.config.lpy_rng.normal(loc=mean_length_growth, scale=0.1 * mean_length_growth)
        return length_incr


@dataclass
class InfoState:
    """Information/metadata for a wood object."""

    age: int = 0
    cut: bool = False
    prunable: bool = True
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
    """ Configuration parameters for BasicWood initialization.
    During one LPy interation the branch will grow some amount along a guide curve (if tied) or along
      a starting direction, curving up.
    The intent is to mimic the tie/prune cycle, so growth parameters are given in terms of yearly growth
    Given a fixed number of iterations per year, we can determine the rest of the parameters
    Buds for cherries are (usually) clearly vegetative versus fruiting, apples are a mix

    For cherrys/apples (most fruit trees) growth in length and angle and bud placement can be characterized
      as follows:
      - Amount a branch will grow in a given year (usually more in the first 1-3 years)
      - Bud spacing along the branch (how far apart are buds, on average?)
      - Bud angle wrt the parent branch - this can vary by bud type (vegetative or fruit) and
      - Bud angle on the branch - spiral pattern, 2/5 phyllotaxis, ie, the next bud will be a rotation of
         144 degrees from the last one
      - Branch diameter vs length is usually related, and follows
         Ideal: < 2.5cm diameter per meter of length
         Vigorous: approaching 3cm / m of length
    [Note - these assume a dwarfing root stock of some sort]

    Given parameters:
      - number of iterations that equal one year
      - A list of yearly growth length ranges by age eg, 12-36" in first year, 12-24" in 3rd year
      - Average bud spacing
      - Phyllotaxis angle
      - Bud angle range by type [vegetative versus fruiting, versus mixed]
      - Overall vigor - what percentage of the branches should be given 'vigorous' growth values
      - Overall curviness of branches (used to create guide curves)
      - What type of tying to do (if any)
      - If the branch is prunable

      - Note: Stopping (or slowing growth) can be handled by having the last yearly_growth_range be really small
    Derived parameters
      - How much to grow by at each iteration
      - Diameters (initial) and diameter growth rates

    Notes on the LString:
    Within the lpy string creation, specifically grow_object, whenever a new bud is started (should_bud
       returned true) then lpy inserts a parameter with the branch and the number of buds (set to zero)
       Should bud defines a segment length to be total"""

    class BudType(Enum):
        VEGETATIVE = "vegetative"
        FRUITING = "fruiting"
        MIXED = "mixed"
        dead = "dead"

    copy_from: any = None
    bud_spacing_range: tuple = (0.0254, 0.0508)  # 1-2 inches
    yearly_growth_range: list[tuple] = None      # eg (1, 24, 36) would be up to 1 year between 24 and 36 inches
    phyllotaxis_angle: float = 144               # How to space buds around a branch
    bud_angle = dict = None                      # Bud angle relative to branch; ngle may depend on type of bud
    tie_type: TyingState.TyingType = TyingState.TyingType.NO_TIE # How to tie this branch type to support
    prunable: bool = True
    name: str = None

    # Random number to use - this is here for repeatability
    lpy_rng: np.random.Generator = None

    # Curve parameters for L-System growth guides
    #   Since growth curves are always in the heading direction (0,0,1) wiggle in x and y but straight in z
    curve_x_range: tuple = (-0.25, 0.25)  # X bounds for Bezier curve control points
    curve_y_range: tuple = (-0.25, 0.25)  # Y bounds for Bezier curve control points
    curve_z_range: tuple = (0, 1)  # Z bounds for Bezier curve control points

    def __post_init__(self):
        """Validate geometric parameters for consistent growth behavior."""
        if self.yearly_growth_range is None:
            """ Set to 24-36 inches the first 3 years, 12-24 for the next 3, then 2-6"""
            self.yearly_growth_range = []
            for _ in range(0, 3):
                self.yearly_growth_range.append((0.3, 0.6))
            self.yearly_growth_range.append((0.05, 0.1))
        if self.bud_angle is None:
            self.bud_angle = {"vegetative": (15, 35),
                              "fruiting": (30-50),
                              "mixed": (15-50)}


class BasicWood(ABC):
    @staticmethod
    def clone(obj):
        try:
            return copy.deepcopy(obj)
        except copy.Error:
            raise copy.Error(f"Not able to copy {obj}") from None

    def __init__(self, config:BasicWoodConfig=None, copy_from:BasicWoodConfig=None, **kwargs):

        # This will be over-riden with either the copy from or input config
        self.config: BasicWoodConfig() = None

        if copy_from is None and config is None:
            raise ValueError("Either 'config' or 'copy_from' must be provided")

        # Handle config-based initialization
        if copy_from:
            self.__copy_constructor__(copy_from)
            return

        if not isinstance(config, BasicWoodConfig):
            raise ValueError("config must be provided when copy_from is None")

        self.location = LocationState()
        # Tying status - contains what type of tying and current tie state and points to tie to
        self.tying = TyingState(tie_type=config.tie_type)

        # Growth Variables  - generate variables stochastically
        vigor = self.config.lpy_rng.normal(0.5, 0.2)
        if vigor < 0.0:
            vigor = 0.0
        if vigor > 1.0:
            vigor = 1.0

        mgl = []
        for mgl_item in self.yearly_growth_range:
            mean_range = (1.0 - vigor) * mgl_item[1] + vigor * mgl_item[2]
            mean_sd = 0.2 * (mgl_item[2] - mgl_item[1])
            mgl = self.config.lpy_rng.normal(mean_range, mean_sd)
            mgl.append(mgl_item[0], mgl_it)
        self.growth = GrowthState(vigour_level=vigor,
                                  config=self.config,

                                  )

        vigour_level: float = 0.5  # Between 0 and 1
        length: float = 0.0
        config: SimulationConfig = None  # Random number generator and number of iteration steps are in here

        # These will be set to values based on vigour and default
        mean_length_growth_per_year: list[float] = None
        thickness: float = 0.001
        mean_thickness_growth_per_iteration: float = 0.1
        target_thickness_ratio = 0.025 / 1.0  # Ideal

        # For tracking growth; year will increment when a year completes
        year: int = 0
        current_iteration: int = 0
        # Information Variables
        self.info = InfoState(order=config.order, color=config.color, material=config.material, prunable=config.prunable)
        # Bud spacing for L-System rules
        self.bud_spacing_age = config.bud_spacing_age

        # Curve parameters for L-System growth guides
        self.curve_x_range = config.curve_x_range
        self.curve_y_range = config.curve_y_range
        self.curve_z_range = config.curve_z_range

        self.logger = logging.getLogger(__name__)

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
        # TODO make stochastic
        self.length += self.growth.growth_length
        self.grow()

    def _find_next_wire_pt(self, end_pt: np.array, end_dir: np.array):
        """ If the end of the branch has gone past the last tie point then find the next wire point
        Also checks that the branch is currently growing in the direction of the wire..."""

        # Vector from branch start to next wire point
        wire_dir = self.tying.wire_attach.attractor_dir
        print(f"Branch {self.name}, indx {self.tying.last_tie_index}", end="")
        if np.dot(wire_dir, end_dir) < 0.0:
            # Oops, branch growing in the wrong direction - set to next wire point to enable reasonable
            # bending at tie down
            next_indx = self.tying.last_tie_index + 1
            if next_indx >= self.tying.wire_attach.attractor_pts.shape[1] - 1:
                # off the end - return -1
                print(" -1")
                return -1
            print(f" {next_indx}")
            return next_indx

        for indx in range(self.tying.last_tie_index + 1, self.tying.wire_attach.attractor_pts.shape[0]):
            wire_point = self.tying.wire_attach.attractor_pts[indx, :]
            v = wire_point - end_pt
            if np.dot(v, wire_dir) >= 0.0:
                print(f" {indx}")
                return indx

        # Off the end of the wire
        print(" -1")
        return -1

    def _get_x_wire_noise(self, pt_branch: np.array, tie_point: np.array):
        # Maximum allowable slide
        dx_max_deviation = self.tying.wire_attach.spacing * 0.2
        x_noisy = pt_branch[0] + np.random.uniform(-dx_max_deviation, dx_max_deviation)
        if x_noisy < tie_point[0] - dx_max_deviation:
            x_noisy = tie_point[0] - dx_max_deviation
        if x_noisy > tie_point[0] + dx_max_deviation:
            x_noisy = tie_point[0] + dx_max_deviation
        return x_noisy

    def _start_guide_points(self):
        """ Find the first tie point that is feasible to reach to and generate a set of guide points to that point
        Use beam deflection to get the shape of the curve
        If tying along then constrain all 3 axes
        If tying across then only constrain y and z"""
        indx_start = self._find_next_wire_pt(end_pt=np.array(self.location.end), end_dir=np.array(self.location.end_dir))
        if indx_start == -1:
            self.logger.info(f"Starting tie down; off end of wire {self.location.end} {self.tying.wire_attach.attractor_pts}")
            indx_start = 0
        self.tying.last_tie_index = indx_start

        start_pt = np.array(self.location.start)
        tie_point = self.tying.wire_attach.attractor_pts[indx_start]
        if self.tying.tie_type == TyingState.TyingType.TIE_ACROSS:
            # Generate a point that is on the wire, but slides a bit on x
            tie_point[0] = self._get_x_wire_noise(pt_branch=start_pt, tie_point=tie_point)

        # Set the guide points to be the deflected curve
        self.tying.guide_points = self._generate_deflected_curve(start_pt, np.array(self.location.end), tie_point)

    def _add_next_guide_points(self):
        """If the end of the branch extends past the last tie point, add some more points to get to the next attractor point."""
        end_dir = np.array(self.location.end_dir)
        indx_start = self._find_next_wire_pt(end_pt=np.array(self.location.end), end_dir=end_dir)
        if indx_start == -1:
            self.logger.info(f"Continuing tie down; off end of wire {self.location.end} {self.tying.wire_attach.attractor_pts}")
            self.tying.last_tie_index = self.tying.wire_attach.attractor_pts.shape[0]
            return
        self.tying.last_tie_index = indx_start

        end_pt = np.array(self.location.end)
        tie_point = self.tying.wire_attach.attractor_pts[indx_start]
        dx = 0.0
        dy = 0.0
        # assuming bent mostly to wire - generate a wriggly curve - determine which side off
        if self.tying.tie_type == TyingState.TyingType.TIE_ACROSS:
            # Generate a point that is on the wire, but slides a bit on x
            tie_point[0] = self._get_x_wire_noise(pt_branch=end_pt, tie_point=tie_point)
            if end_dir[0] < 0.0:
                dx = -np.abs(np.random.normal(loc=0.0, scale=0.5*self.tying.wire_attach.spacing))
            else:
                dx = np.abs(np.random.normal(loc=0.0, scale=0.5 * self.tying.wire_attach.spacing))
        else:
            if end_dir[1] < 0.0:
                dy = -np.abs(np.random.normal(loc=0.0, scale=0.5*self.tying.wire_attach.spacing))
            else:
                dy = np.abs(np.random.normal(loc=0.0, scale=0.5 * self.tying.wire_attach.spacing))

        dt = 1.0 / 4.0  # Add two points to the middle between last guide point and tie point
        last_guide_point = np.array(self.tying.guide_points[-1])
        for indx in range(1, 3):
            t = dt * indx
            interpolate_wire_pt = (1.0 - t) * last_guide_point + t * tie_point
            interpolate_wire_pt[0] += dx
            interpolate_wire_pt[1] += dy
            self.tying.guide_points.append(tuple(interpolate_wire_pt))
        self.tying.guide_points.append(tuple(tie_point))

    @staticmethod
    def _deflection_at_x(d, x, L):
        """d is the max deflection, x is the current location we need deflection on and L is the total length"""
        return (d / 2) * (x**2) / (L**3) * (3 * L - x)

    def _generate_deflected_curve(self, start_pt: np.array, end_pt: np.array, tie_point: np.array):
        control_points = []
        deflection_vector = tie_point - end_pt
        branch_length = np.linalg.norm(end_pt - start_pt)
        if np.isclose(branch_length, 0.0):
            branch_length = np.linalg.norm(tie_point - start_pt)
        # Parametric position along branch segment [0.1, 0.2, ..., 1.0]
        for t in np.arange(0.1, 1.1, 0.1):
            # Base position: linear interpolation from start to current
            base_position = start_pt + t * (end_pt - start_pt)

            # Add beam deflection (cantilever formula)
            deflection = self._deflection_at_x(deflection_vector, t * branch_length, branch_length)

            # Combine base position and deflection
            point = tuple(base_position + deflection)
            control_points.append(point)
        return control_points

    @abstractmethod
    def create_branch(self):
        """Returns how a new order branch when bud break happens will look like if a bud break happens"""
        pass
        # new_object = BasicWood.clone(self.branch_object)
        # return new_object
        # return BasicWood(self.num_buds_segment/2, self.bud_break_prob, self.thickness/2, self.thickness_increment/2, self.growth_length/2,\
        # self.max_length/2, self.tie_type, self.bud_break_max_length/2, self.order+1, self.bud_break_prob_func)

    def update_guide(self):
        """ If the branch/trunk has grown past the last tie point then append more guide points
        Also updates the tying variables (last tie point, guide points, guide_length)

        Notes:
            - If out of tie points sets guide_length to be zero (no longer follow curve)
            - Appends guide points incrementally to self.tying.guide_points
            - Only generates a new set of guide points when the branch crosses a tying point
            - Uses self.location.start as base if not yet tied; the last tie point in WireBranchAttach otherwise.
            - Adds some stochasticity to the guide curve by 1) letting across tie points slide in x and 2) following the
               direction of the branch growth (curve 'bows' out of guide)
        """
        if not self.tying.is_tied:
            return

        self.logger.info(f"Updating guide {self.name} {self.tying.last_tie_index}")
        # Ran out of tie points
        if self.tying.last_tie_index >= self.tying.wire_attach.attractor_pts.shape[1]:
            self.logger.info(f"  Off end")
            return

        # Case 1: We haven't started tying yet, so create a guide curve that goes from the end point
        #         to the first tie point
        # Case 2: We are still growing along the guide curve, haven't reached the end
        # Case 3: We have grown past the last tie point and need to add to the guide curve
        # Sets self.tying.guide_points and self.tying.guide_length
        self.tying.tie_needs_updating = False
        if self.tying.last_tie_index == -1:
            self._start_guide_points()
            # Flag that we need to update the guide curve in the LString
            self.tying.tie_needs_updating = True
        else:
            # check location of end point wrt last tie point
            # The last guide point will have been set to the last tie point
            end_pt = np.array(self.location.end)
            last_tie_pt = self.tying.wire_attach.attractor_pts[self.tying.last_tie_index]
            dir_along = end_pt - last_tie_pt
            past = np.dot(dir_along, self.tying.wire_attach.attractor_dir)
            if past > 0.0:
                self._add_next_guide_points()
                # Flag that we need to update the guide curve in the LString
                self.tying.tie_needs_updating = True
        # Note: actual guide curve will be updated at EndEach hook, not here
        self.logger.info(f"Done updating guide {self.name} {self.tying.last_tie_index} {self.tying.guide_points[-1]}")

    def tie_lstring(self, lstring, index):
        """Insert a SetGuide(...) after position `index` in `lstring`.

        - Removes any immediate following tokens whose .name is in ('&','/','SetGuide').
        - Builds a CSpline from `self.tying.guide_points` and inserts the curve string and length.
        Returns (lstring, removed_count).
        """
        # Nothing to do if we don't have new guide points
        if not self.tying.tie_needs_updating:
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

        # Insert the new SetGuide token at the computed insert position
        # Upper limit on following the guide curve is the number of wires currently crossed * spacing
        # Once the turtle hits the set guide it will follow it for the length given; so give it the length
        #   of the guide curve
        tie_length = 1.1 * self.tying.last_tie_index * self.tying.wire_attach.spacing
        lstring.insertAt(insert_pos, f"SetGuide({curve_repr}, {tie_length})")

        # Flag that we've added the new guide curve
        self.tying.tie_needs_updating = False

        return lstring, removed_count

    def _find_wire_pt(self, targets: list[tuple], start: tuple, current: tuple):
        """ Convert to numpy and find the next wire point"""
        start_arr = np.array([start[0], start[1], start[2]], dtype=float)
        current_arr = np.array([current[0], current[1], current[2]], dtype=float)
        wire_points = np.array(targets, dtype=float)
        wire_axis = np.array(wire_points[-1, :] - wire_points[0, :])
        len_wire_axis = np.linalg.norm(wire_axis)
        if len_wire_axis > 0.0:
            wire_axis = wire_axis / len_wire_axis
        else:
            # Shouldn't happen - but assume wire is in x direction
            wire_axis = np.array([1, 0, 0])

        # Vector from branch start to next wire point
        v = np.zeros((1, 3))
        wire_point = wire_points[0, :]
        for try_point in range(0, wire_points.shape[0]):
            wire_point = wire_points[try_point, :]
            v = wire_point - start_arr
            if np.dot(v, wire_axis) >= 0.0:
                break

        return start_arr, current_arr, wire_point, wire_axis, v

    def _get_control_pts_along(self, targets: list[tuple], start: tuple, current: tuple):
        """ just offset in the y direction"""
        start_arr, current_arr, wire_point, wire_axis, _ = self._find_wire_pt(targets=targets, start=start, current=current)

        control_points = []
        est_length = np.linalg.norm(wire_point - start_arr)
        for step in np.arange(-est_length, 0.1, 0.1):
            pt_wire = wire_point + wire_axis * step
            pt_wire[0] = start_arr[0]
            control_points.append(tuple(pt_wire))
        return control_points

    def get_control_points(self, targets : list[tuple], start: tuple, current: tuple, tie_type: TyingState.TyingType):
        """
        Compute control points for a 3D curve from branch segment to tie point on wire.

        Uses vector projection to determine feasibility and compute the tie point location,
        then generates a deflected curve using beam theory.

        Args:
            targets: Wire points list of (x, y, z) points on the wire
            start: Branch segment start point (x, y, z)
            current: Branch segment end point (x, y, z)
            tie_type: Either along or across the wire

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
        if tie_type == TyingState.TyingType.TIE_ACROSS:
            return self._get_control_pts_along(targets=targets, start=start, current=current)

        # Convert inputs to numpy arrays
        start_arr, current_arr, wire_point, wire_axis, v = self._find_wire_pt(targets=targets, start=start, current=current)

        # Calculate branch segment length
        segment_vector = current_arr - start_arr
        branch_length = np.linalg.norm(segment_vector)
        if branch_length < BasicWood.eps:
            return [], None  # Degenerate segment

        if np.dot(v, wire_axis) < 0.0:
            # No target wire point past the end of the branch - quit pinning
            return [], None  # Degenerate segment

        # Decompose v into components parallel and perpendicular to wire axis
        parallel_component, perpendicular_component = self._get_parallel_and_perpendicular_components(v, wire_axis)
        perpendicular_distance = np.linalg.norm(perpendicular_component)

        # Feasibility check: branch must be long enough to reach the wire
        if perpendicular_distance > branch_length:
            return [], None

        # Calculate distance to travel along wire (Pythagorean theorem)
        # branch_length² = perpendicular_distance² + parallel_travel²
        parallel_travel_sq = branch_length**2 - perpendicular_distance**2
        if parallel_travel_sq < 0.0:
            parallel_travel_sq = 0.0
        parallel_travel = np.sqrt(parallel_travel_sq)  # Clamp to avoid floating-point negatives

        # Compute tie point on wire
        # Start from perpendicular projection of start onto wire, then move parallel_travel along wire
        start_projection_on_wire = start_arr + perpendicular_component
        direction_to_wire = np.sign(np.dot(wire_point, wire_axis))
        tie_point = start_projection_on_wire + parallel_travel * wire_axis * direction_to_wire

        # Generate control points along deflected curve using beam deflection formula
        control_points = self._generate_deflected_curve(start_arr, current_arr, tie_point)

        return control_points, tuple(tie_point)

    def _get_parallel_and_perpendicular_components(self, vec_a, vec_b):
        # Project vec_a onto vec_b to get parallel and perpendicular components
        vec_b_unit = vec_b / np.linalg.norm(vec_b)
        parallel_component = np.dot(vec_a, vec_b_unit) * vec_b_unit
        perpendicular_component = vec_a - parallel_component
        return parallel_component, perpendicular_component


class TreeBranch(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    count = 0    # Class variable for instance counting

    def __init__(self,
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
