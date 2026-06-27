"""
Defines type of tying
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R
from lpy_treesim.tree_models.base_tree.basic_wood_growth_location import LocationState
from openalea.plantgl.all import Vector3, Vector4


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

    # Shouldn't have to change these, but making variables just in case
    n_pts_tie_down: int = 8
    n_pts_per_tie: int = 4

    wire_attach: WireBranchAttach = None  # These are the list of points to tie to
    # Save the last starting point and direction of tying -if these change, need to re-do
    start_pt_tied: tuple = (100, 100, 100)
    start_dir_tied: tuple = (100, 100, 100)
    # Potential support structure info - used to create guide_point spacing
    #  These should be specified in the tree wood configuration files because they are trunk/branch dependent
    tie_start_dist: float = 0.46  # 18 inches Expected distance from the base of the trunk/branch to the first tie point
    tie_spacing: float = 0.46     # 18 inches Spacing between tie points

    # Guide points are relative to branch's base coordinate system
    guide_points: list[Vector4] = None  # Current set of guide points (control points of Bezier curve)
    tie_type: TyingType = TyingType.NO_TIE  # One of tying_type

    @property
    def is_tied(self):
        return self.wire_attach is not None

    def has_moved(self, start_pt:tuple, start_dir: tuple):
        """ Check against current"""
        for indx in range(0, 3):
            if not np.isclose(start_pt[indx], self.start_pt_tied[indx]):
                return True
            if not np.isclose(start_dir[indx], self.start_dir_tied[indx]):
                return True
        return False

    def __post_init__(self):
        """Initialize guide_points as empty list if not provided."""
        if self.guide_points is None:
            self.guide_points = []

    def create_initial_guide_curve(self,
                                   expected_length: float,
                                   curve_x_range: tuple,
                                   curve_y_range: tuple,
                                   lpy_rng: np.random.Generator):
        """If not tying, just a curve with some noise in x and y and evenly spaced points. Otherwise,
        Set up so that first n points are movable (deform when tied) and there are 3 points between
        tie points"""
        self.guide_points = []
        if self.tie_type == TyingState.TyingType.NO_TIE:
            z_values = np.linspace(0.0, expected_length, 6)
        else:
            n_tie_regions = int((expected_length - self.tie_start_dist) / self.tie_spacing)
            # number of tie regions should be correct if the tie spacing distance and prune length are set correctly
            if n_tie_regions < 6:
                n_tie_regions = 6  # Just in case not enough
            z_values = np.linspace(0.0, self.tie_start_dist, self.n_pts_tie_down)
            spacing_values = np.linspace(0.0, self.tie_spacing, self.n_pts_per_tie+1)
            for indx in range(0, n_tie_regions):
                spacing_values_shifted = spacing_values + z_values[-1]
                z_values = np.concatenate((z_values, spacing_values_shifted[1:]))
            # Add one more - last tied down point
            stupid_numpy = np.zeros((z_values.shape[0] + 1, ))
            stupid_numpy[0:-1] = z_values
            stupid_numpy[-1] = z_values[-1] + spacing_values[1]
            z_values = stupid_numpy

        # start at origin
        self.guide_points.append(Vector4(0.0, 0.0, 0.0, 1))
        # Take out that z value
        z_values = z_values[1:]
        for z_value in z_values:
            x_coord = lpy_rng.uniform(curve_x_range[0], curve_x_range[1])
            y_coord = lpy_rng.uniform(curve_y_range[0], curve_y_range[1])
            # Uncomment these if you want to take the noise out of branches (for debugging)
            x_coord = 0
            y_coord = 0
            self.guide_points.append(Vector4(x_coord, y_coord, z_value, 1))
        if self.tie_type == TyingState.TyingType.NO_TIE:
            return

        # Smooth out the points inbetween (the extra points)
        #  Pin points are at
        #         0
        #         n_pts_tie_down
        #         every n_pts_per_tie after that (which is why there's one extra)
        n_pts_smooth = self.n_pts_tie_down
        indx_start = 0
        while indx_start < len(self.guide_points) - 1:
            pt_start = self.guide_points[indx_start]
            pt_mid = self.guide_points[indx_start + n_pts_smooth // 2]
            pt_end = self.guide_points[indx_start + n_pts_smooth]
            # print(f"start {pt_start} mid {pt_mid} end {pt_end}")
            mat_solve_a = np.ones((3, 3))
            mat_solve_b = np.zeros((3, 2))
            for indx, pt in enumerate((pt_start, pt_mid, pt_end)):
                mat_solve_a[indx, 0] = pt[2] * pt[2]  # the t value
                mat_solve_a[indx, 1] = pt[2]  # the t value
                mat_solve_b[indx, 0] = pt[0]
                mat_solve_b[indx, 1] = pt[1]
            ls = np.linalg.lstsq(mat_solve_a, mat_solve_b)
            x = ls[0]
            for indx in range(0, n_pts_smooth):
                z = self.guide_points[indx_start + indx][2]
                pt_x = x[0, 0] * z * z + x[1, 0] * z + x[2, 0]
                pt_y = x[0, 1] * z * z + x[1, 1] * z + x[2, 1]

                # print(f" Before {self.guide_points[indx_start + indx]}")
                self.guide_points[indx_start + indx][0] = pt_x
                self.guide_points[indx_start + indx][1] = pt_y
                # print(f" After {self.guide_points[indx_start + indx]}")
            # Now smooth out between tie points
            indx_start += n_pts_smooth
            n_pts_smooth = self.n_pts_per_tie
            # print("\n")

        # for pt in self.guide_points:
        #     print(f"{pt}")

    def _convert_guide_points_to_global(self, pt_origin: Vector3, heading: Vector3, left: Vector3):
        """ Convert guide points from base curve position to global"""
        gp_as_np = np.array(self.guide_points)
        rot_mat = np.identity(3)
        # Heading goes in z, left goes in y, x is the other one
        rot_mat[1, :] = np.array(left)
        rot_mat[2, :] = np.array(heading)
        rot_mat[0, :] = np.cross(rot_mat[1, :], rot_mat[2, :])
        for ir in range(gp_as_np.shape[0]):
            for ic in range(0, 3):
                # Guide curves start at 0,0,0 - move to branch start point in space
                gp_as_np[ir, ic] += pt_origin[ic]
            gp_as_np[ir, 0:3] = rot_mat @ gp_as_np[ir, 0:3]

        vec = gp_as_np[1, :] - gp_as_np[0, :]
        vec = vec / np.linalg.norm(vec)
        print(f"vec {vec} heading {heading}")
        return rot_mat, gp_as_np

    def _find_pt_on_wire(self, start_pt, next_guide_pt, next_wire_pt):
        """ Rather than pin to the exact wire point, let it 'slide' along the wire"""
        vec_to_guide_point = next_guide_pt - start_pt
        vec_to_wire_point = next_wire_pt - start_pt
        dist_along_wire_guide = np.dot(vec_to_guide_point, self.wire_attach.attractor_dir)
        dist_allow_slide = self.wire_attach.spacing * 0.5
        d_slide = dist_allow_slide * np.tanh(dist_along_wire_guide)
        d_slide = 0.0
        return next_wire_pt + self.wire_attach.attractor_dir * d_slide

    def _angle_between(self, vec_from : np.array, vec_to: np.array):
        """ Input are 2d vectors; normalize, and find angle using acos"""
        # Rotate the first vector to (1, 0)
        ang = np.atan2(vec_from[1], vec_from[0])
        mat_rot = R.from_euler('z', ang).as_matrix()
        vec_from_rot = mat_rot[0:2, 0:2] @ vec_from
        vec_to_rot = mat_rot[0:2, 0:2] @ vec_to
        ang_rot_to = np.atan2(vec_to_rot[1], vec_to_rot[0])
        # Check
        mat_rot_check = R.from_euler('z', ang_rot_to).as_matrix()
        vec_rot_check = mat_rot[0:2, 0:2] @ vec_to
        print(f"Vec from {vec_from} vec to {vec_rot_check}")
        return ang_rot_to

    def _get_angs_and_scl(self, start_pt, next_guide_pt, next_wire_pt):
        """ What are the rotation angles and scale needed to move the selected guide point to the next wire point?"""
        vec_to_guide_point = next_guide_pt - start_pt
        vec_to_wire_point = next_wire_pt - start_pt
        print(f"Vec guide {vec_to_guide_point}, Vec wire {vec_to_wire_point}")

        len_to_wire = np.linalg.norm(vec_to_wire_point)
        len_to_guide = np.linalg.norm(vec_to_guide_point)

        if np.isclose(len_to_wire, 0.0):
            return 1.0, vec_to_guide_point, 0.0

        scl = len_to_wire / len_to_guide
        vec_to_guide_point *= scl
        len_new_guide = np.linalg.norm(vec_to_guide_point)
        print(f"Old length {len_to_guide} wanted {len_to_wire} now {len_new_guide}")

        vec_to_guide_point = vec_to_guide_point / len_to_wire
        vec_to_wire_point = vec_to_wire_point / len_to_wire
        dot = np.dot(vec_to_guide_point, vec_to_wire_point)
        if np.isclose(dot, 1.0):
            return scl, vec_to_guide_point, 0.0

        ang = np.acos(dot)
        vec_cross = np.cross(vec_to_guide_point, vec_to_wire_point)
        vec_cross = vec_cross / np.linalg.norm(vec_cross)
        return scl, vec_cross, ang

    def _bend_to_wire(self, pt_origin: Vector3, heading: Vector3, left: Vector3):
        """
        Convert points to global coords, then do a pivot around each point in turn to incrementally align
        the nth point with the wire guide point
        Note: need to re-do if the initial point/vector change
        """
        rot_mat, gp_as_np = self._convert_guide_points_to_global(pt_origin=pt_origin, heading=heading, left=left)
        start_indx = 0
        n_spacing = self.n_pts_tie_down
        indx_guide_pts = [start_indx]
        for tie_point in range(0, self.wire_attach.attractor_pts.shape[0]):
            # Pivoting around indx point to bring the next tie point to the next wire point
            start_pt = gp_as_np[start_indx, 0:3]
            next_guide_pt = gp_as_np[start_indx + n_spacing, 0:3]  # start of next tie down group
            next_wire_pt = self.wire_attach.attractor_pts[tie_point, 0:3]
            if self.tie_type == TyingState.TyingType.TIE_ACROSS:
                next_wire_pt = self._find_pt_on_wire(start_pt=start_pt,
                                                     next_guide_pt=next_guide_pt,
                                                     next_wire_pt=next_wire_pt)
            scl, vec, ang = self._get_angs_and_scl(start_pt=start_pt,
                                                   next_guide_pt=next_guide_pt,
                                                   next_wire_pt=next_wire_pt)
            print(f"Tie point start {start_pt} wire {next_wire_pt} guide_point {next_guide_pt}")
            print(f"Scl {scl} Vec {vec}, ang {ang}")
            # Now rotate/scale all of the guide curve to the right, using some percentage of the rotate scale up to the
            # next index point
            # Scale all points uniformly
            mat_scl = np.identity(3)
            mat_scl[0, 0] = scl
            mat_scl[1, 1] = scl
            mat_scl[2, 2] = scl
            mat_rot = R.from_rotvec(vec * ang).as_matrix()
            # Move ALL the points after this one by the target amount
            for pt_indx in range(start_indx+1, gp_as_np.shape[0]):
                if pt_indx - start_indx < n_spacing:
                    perc_along = (pt_indx - start_indx) / (n_spacing - 1.0)
                    mat_rot = R.from_rotvec(vec * ang * perc_along).as_matrix()
                # Subtract the current point (the one we're rotating around)
                pt = gp_as_np[pt_indx, 0:3] - start_pt
                # Do the rotation
                pt_rot = mat_scl @ mat_rot @ pt
                # Translate back out
                pt_back = pt_rot + start_pt
                gp_as_np[pt_indx, 0:3] = pt_back

            # This *should* be the same as the next wire point
            new_next_guide = gp_as_np[start_indx + n_spacing, 0:3]
            vec_guide = new_next_guide - start_pt
            vec_guide = vec_guide / np.linalg.norm(vec_guide)
            vec_wire = next_wire_pt - start_pt
            vec_wire = vec_wire / np.linalg.norm(vec_wire)
            print(f"  Vector dot {np.dot(vec_guide, vec_wire)} New guide point loc {gp_as_np[start_indx + n_spacing, 0:3]}")
            start_indx += n_spacing
            n_spacing = self.n_pts_per_tie
            indx_guide_pts.append(start_indx)
        print(f"Guide points", end="")
        for indx in indx_guide_pts:
            print(f"{gp_as_np[indx, 0:3]} ")
        print(f"Wire points\n{self.wire_attach.attractor_pts}")
        print("done")
        self.guide_points = []
        for indx in range(0, gp_as_np.shape[0]):
            # Put back at the branch location, but use global orientation
            # LPy will add a @R to reset orientation to the default
            pt = gp_as_np[indx, 0:3] + np.array(pt_origin)
            self.guide_points.append(Vector4(pt[0], pt[1], pt[2], 1.0))
        """
        # Convert the guide points back to local coordinate system
        for icoord in range(0, 3):
            gp_as_np[:, icoord] -= pt_origin[icoord]
        rot_mat_back = rot_mat.transpose()
        for indx in range(0, gp_as_np.shape[0]):
            rot_vec = rot_mat_back @ gp_as_np[indx, 0:3]
            self.guide_points.append(Vector4(rot_vec[0], rot_vec[1], rot_vec[2], 1.0))
        """

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


        # Vector from branch start to next wire point
        wire_dir = self.wire_attach.attractor_dir
        print(f" indx {self.last_tie_index}", end="")
        end_dir = np.ones((3, 1))
        end_pt = np.ones((3, 1))
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


