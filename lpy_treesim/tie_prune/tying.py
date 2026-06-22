"""
Defines type of tying
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from lpy_treesim.tree_models.base_tree.basic_wood_growth_location import LocationState
from openalea.plantgl.all import Vector3


@dataclass
class WireBranchAttach:
    # where a branch is attached to a wire
    x_along: float = 0.0
    branch_id: int = -1
    attractor_pts: np.array = None
    attractor_dir: np.array = None
    spacing: float = 0.0


@dataclass
class TyingState:
    """Tying and guiding state for a wood object."""
    class TyingType(Enum):
        NO_TIE = "no_tie"
        TIE_ALONG = "tie_along"
        TIE_ALONG_FIRST = "tie_along_first"
        TIE_ACROSS = "tie_across"

    tie_needs_updating: bool = False  # Set to false when guide curve updated, true when branch changes/new guide point added
    wire_attach: WireBranchAttach = None  # These are the list of points to tie to
    last_tie_index: int = -1          # Branch has been tied to all of the points up to this index
    # Guide points are relative to branch's base coordinate system
    guide_points: list[tuple] = None  # Current set of guide points (control points of Bezier curve)
    tie_type: TyingType = TyingType.NO_TIE  # One of tying_type

    @property
    def is_tied(self):
        return self.wire_attach is not None

    def __post_init__(self):
        """Initialize guide_points as empty list if not provided."""
        if self.guide_points is None:
            self.guide_points = []

    def set_initial_guide_points(self,
                                 x_range=(-2, 2),
                                 y_range=(-2, 2),
                                 total_length=10.0,
                                 rng: np.random.Generator=None):
    """ Create Bezier curve guide points using wire spacing as a guide.
    The guide points are in local coordinates; the intent is to have guide points spaced by wire spacing.
    z is the heading direction, x and y are noise in and out/left-right """
    if rng is None:
        rng = np.random.default_rng()
    if self.wire_attach:
        spacing =
    # Generate control points with progressive z-coordinates
    z_values = np.linspace(0.0, total_length, num_control_points)
    control_points = []

    for z_value in z_values:
        x_coord = rng.uniform(x_range[0], x_range[1])
        y_coord = rng.uniform(y_range[0], y_range[1])
        control_points.append(Vector4(x_coord, y_coord, z_value, 1))

    # Create PlantGL Bezier curve
    control_point_array = Point4Array(control_points)
    return BezierCurve(control_point_array)

    def _convert_guide_points_to_global(self, pt_origin: Vector3, heading: Vector3, left: Vector3 ):
        """ Convert guide points from base curve position to global"""
        gp_as_np = np.array(self.guide_points)
        rot_mat = np.identity(3)
        rot_mat[0, :] = np.array(heading)
        rot_mat[1, :] = np.array(left)
        rot_mat[2, :] = np.cross(rot_mat[0, :], rot_mat[1, :])
        for ir in range(gp_as_np.shape[0]):
            for ic in range(0, 3):
                gp_as_np[ir, ic] -= pt_origin[ic]
                gp_as_np[ir, :] = rot_mat @ gp_as_np[ir, :]
        return gp_as_np

    def _lengths_guide_pts(self, gps: np.array):
        """ Spacing between guide points"""
        lengths = []
        for ir in range(0, gps.shape[0] - 1):
            lengths.append(np.linalg.norm(gps.shape[ir+1, :] - gps.shape[ir, :]))
        return lengths

    def _assign_pts_wire(self, start_pt: np.array, lengths_cntrl_pts: list[float], gps: np.array):
        """ Assign each guide point a wire index and t value between. Assign -1 if before wire or n if after
        Assumption is that the starting point of the branch """
        start_indx = 0
        start_dist = 0.0
        wire_dir = self.wire_attach.attractor_dir
        for indx in range(0, self.wire_attach.attractor_pts.shape[0]):
            dir_to_wire = self.wire_attach.attractor_pts[indx, :] - start_pt
            if np.dot(wire_dir, dir_to_wire) > 0.0:
                start_indx = indx
                start_dist = np.linalg.norm(dir_to_wire)
                break

        assignment = []
        last_indx = start_indx
        for indx in range(0, gps.shape[0]):
            if lengths_cntrl_pts[indx] < start_dist:
                assignment.append((-1, 1.0))
            elif lengths_cntrl_pts[indx]


        # Vector from branch start to next wire point
        wire_dir = self.wire_attach.attractor_dir
        print(f" indx {self.last_tie_index}", end="")
        if np.dot(wire_dir, end_dir) < 0.0:
            # Oops, branch growing in the wrong direction - set to next wire point to enable reasonable
            # bending at tie down
            next_indx = self.last_tie_index + 1
            if next_indx >= self.wire_attach.attractor_pts.shape[1] - 1:
                # off the end - return -1
                print(" -1")
                return -1
            print(f" {next_indx}")
            return next_indx

        for indx in range(self.last_tie_index + 1, self.wire_attach.attractor_pts.shape[0]):
            wire_point = self.wire_attach.attractor_pts[indx, :]
            v = wire_point - end_pt
            if np.dot(v, wire_dir) >= 0.0:
                print(f" {indx}")
                return indx

        # Off the end of the wire
        print(" -1")
        return -1

    def _find_next_wire_pt(self, end_pt: np.array, end_dir: np.array):
        """ If the end of the branch has gone past the last tie point then find the next wire point
        Also checks that the branch is currently growing in the direction of the wire..."""

        # Vector from branch start to next wire point
        wire_dir = self.wire_attach.attractor_dir
        print(f" indx {self.last_tie_index}", end="")
        if np.dot(wire_dir, end_dir) < 0.0:
            # Oops, branch growing in the wrong direction - set to next wire point to enable reasonable
            # bending at tie down
            next_indx = self.last_tie_index + 1
            if next_indx >= self.wire_attach.attractor_pts.shape[1] - 1:
                # off the end - return -1
                print(" -1")
                return -1
            print(f" {next_indx}")
            return next_indx

        for indx in range(self.last_tie_index + 1, self.wire_attach.attractor_pts.shape[0]):
            wire_point = self.wire_attach.attractor_pts[indx, :]
            v = wire_point - end_pt
            if np.dot(v, wire_dir) >= 0.0:
                print(f" {indx}")
                return indx

        # Off the end of the wire
        print(" -1")
        return -1

    def _get_x_wire_noise(self, pt_branch: np.array, tie_point: np.array):
        # Maximum allowable slide
        dx_max_deviation = self.wire_attach.spacing * 0.2
        x_noisy = pt_branch[0] + np.random.uniform(-dx_max_deviation, dx_max_deviation)
        if x_noisy < tie_point[0] - dx_max_deviation:
            x_noisy = tie_point[0] - dx_max_deviation
        if x_noisy > tie_point[0] + dx_max_deviation:
            x_noisy = tie_point[0] + dx_max_deviation
        return x_noisy

    def start_guide_points(self, loc: LocationState):
        """ Find the first tie point that is feasible to reach to and generate a set of guide points to that point
        Use beam deflection to get the shape of the curve
        If tying along then constrain all 3 axes
        If tying across then only constrain y and z"""
        indx_start = self._find_next_wire_pt(end_pt=np.array(loc.end), end_dir=np.array(loc.end_dir))
        if indx_start == -1:
            print(f"Starting tie down; off end of wire {loc.end} {self.wire_attach.attractor_pts}")
            indx_start = 0
        self.last_tie_index = indx_start

        start_pt = np.array(loc.start)
        tie_point = self.wire_attach.attractor_pts[indx_start]
        if self.tie_type == TyingState.TyingType.TIE_ACROSS:
            # Generate a point that is on the wire, but slides a bit on x
            tie_point[0] = self._get_x_wire_noise(pt_branch=start_pt, tie_point=tie_point)

        # Set the guide points to be the deflected curve
        self.guide_points = self._generate_deflected_curve(start_pt, np.array(loc.end), tie_point)

    def add_next_guide_points(self, loc: LocationState):
        """If the end of the branch extends past the last tie point, add some more points to get to the next attractor point."""
        end_dir = np.array(loc.end_dir)
        indx_start = self._find_next_wire_pt(end_pt=np.array(loc.end), end_dir=end_dir)
        if indx_start == -1:
            print(f"Continuing tie down; off end of wire {loc.end} {self.wire_attach.attractor_pts}")
            self.last_tie_index = self.wire_attach.attractor_pts.shape[0]
            return
        self.last_tie_index = indx_start

        end_pt = np.array(loc.end)
        tie_point = self.wire_attach.attractor_pts[indx_start]
        dx = 0.0
        dy = 0.0
        # assuming bent mostly to wire - generate a wriggly curve - determine which side off
        if self.tie_type == TyingState.TyingType.TIE_ACROSS:
            # Generate a point that is on the wire, but slides a bit on x
            tie_point[0] = self._get_x_wire_noise(pt_branch=end_pt, tie_point=tie_point)
            if end_dir[0] < 0.0:
                dx = -np.abs(np.random.normal(loc=0.0, scale=0.5*self.wire_attach.spacing))
            else:
                dx = np.abs(np.random.normal(loc=0.0, scale=0.5 * self.wire_attach.spacing))
        else:
            if end_dir[1] < 0.0:
                dy = -np.abs(np.random.normal(loc=0.0, scale=0.5*self.wire_attach.spacing))
            else:
                dy = np.abs(np.random.normal(loc=0.0, scale=0.5 * self.wire_attach.spacing))

        dt = 1.0 / 4.0  # Add two points to the middle between last guide point and tie point
        last_guide_point = np.array(self.guide_points[-1])
        for indx in range(1, 3):
            t = dt * indx
            interpolate_wire_pt = (1.0 - t) * last_guide_point + t * tie_point
            interpolate_wire_pt[0] += dx
            interpolate_wire_pt[1] += dy
            self.guide_points.append(tuple(interpolate_wire_pt))
        self.guide_points.append(tuple(tie_point))

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

    def get_control_points(self, targets : list[tuple], start: tuple, current: tuple, tie_type: TyingType):
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
        if branch_length < 0.0001:
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


