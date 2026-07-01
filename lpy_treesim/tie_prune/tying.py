"""
Defines type of tying
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R
from openalea.plantgl.all import Vector3, Vector4


@dataclass
class WireBranchAttach:
    # where a branch is attached to a wire - Created in make_attractor_grid in wire_support.py
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

    # Shouldn't have to change these, but making variables just in case. This is the number of in-between guide points
    #  between tie downs. Allows for a bit of bending between tie points
    n_pts_tie_down: int = 8
    n_pts_per_tie: int = 4

    wire_attach: WireBranchAttach = None  # These are the list of points to tie to

    # Save the last starting point and direction of tying -if these change, need to re-do. Set in BasicWood
    start_pt_tied: tuple = (100, 100, 100)
    start_dir_tied: tuple = (100, 100, 100)
    # Potential support structure info - used to create guide_point spacing
    #  These should be specified in the tree wood configuration files because they are trunk/branch dependent
    tie_start_dist: float = 0.46  # 18 inches Expected distance from the base of the trunk/branch to the first tie point
    tie_spacing: float = 0.46     # 18 inches Spacing between tie points

    # Guide points are relative to branch's base coordinate system
    #   - Initial guides (not tied down) use the branch's coordinate system (z is the branch growth direction)
    #   - Once tied down, these guide points start at the base of the branch but use the global coordinate system
    #      (z is up) so these points *should* be a translation of the wire attach points
    guide_points: list[Vector4] = None  # Current set of guide points (control points of Bezier curve)
    tie_type: TyingType = TyingType.NO_TIE  # One of tying_type

    @property
    def is_tied(self):
        return self.wire_attach is not None

    def has_moved(self, start_pt:tuple, start_dir: tuple):
        """ Check against current tie point and direction """
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
        """If branch is not a ty-able type, just a curve with some noise in x and y and evenly spaced points. Otherwise,
        set up so that first n points are movable (deform when tied) and there are 3 points between
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
            # x_coord = 0
            # y_coord = 0
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
        """ Convert guide points from base curve position to global, re-orient to global frame"""
        gp_as_np = np.array(self.guide_points)
        rot_mat = np.identity(3)
        # Heading goes in z, left goes in y, x is the other one
        rot_mat[:, 0] = np.array(left)
        rot_mat[:, 2] = np.array(heading)
        rot_mat[:, 1] = np.cross(rot_mat[0, :], rot_mat[2, :])
        pt_curve_origin = np.copy(gp_as_np[0, 0:3])
        # print(f"Pt origin global: {pt_origin}, pt curve local origin {pt_curve_origin}")
        for ir in range(gp_as_np.shape[0]):
            # Rotate around origin (first point in the guide points should be 0,0,0)
            pt = gp_as_np[ir, 0:3] - pt_curve_origin
            gp_as_np[ir, 0:3] = rot_mat @ pt
            for ic in range(0, 3):
                # Guide curves start at 0,0,0 - move to branch start point in space
                gp_as_np[ir, ic] += pt_origin[ic]

        """
        vec = gp_as_np[1, :] - gp_as_np[0, :]
        vec = vec / np.linalg.norm(vec)
        x = np.zeros((3, 1))
        x[0] = 1
        print(f"Heading {heading}, guide points {vec}")
        print(f"Left {left}, guide points {(rot_mat @ x).transpose()}")
        """
        return rot_mat, gp_as_np

    def _find_pt_on_wire(self, start_pt, next_guide_pt, next_wire_pt):
        """ Rather than pin to the exact wire point, let it 'slide' along the wire"""
        vec_to_guide_point = next_guide_pt - start_pt
        dist_along_wire_guide = np.dot(vec_to_guide_point, self.wire_attach.attractor_dir)
        dist_allow_slide = self.wire_attach.spacing * 0.5
        d_slide = dist_allow_slide * np.tanh(dist_along_wire_guide)
        # Uncomment if you want to disable sliding (for debugging)
        #d_slide = 0.0
        return next_wire_pt + self.wire_attach.attractor_dir * d_slide

    def _get_angs_and_scl(self, start_pt, next_guide_pt, next_wire_pt):
        """ What are the rotation angles and scale needed to move the selected guide point to the next wire point?
        Rotation is specified as a rotation around a vector by amount"""
        vec_to_guide_point = next_guide_pt - start_pt
        vec_to_wire_point = next_wire_pt - start_pt
        # print(f"Vec guide {vec_to_guide_point}, Vec wire {vec_to_wire_point}")

        # Scale
        len_to_wire = np.linalg.norm(vec_to_wire_point)
        len_to_guide = np.linalg.norm(vec_to_guide_point)

        if np.isclose(len_to_wire, 0.0):
            return 1.0, vec_to_guide_point, 0.0

        scl = len_to_wire / len_to_guide
        vec_to_guide_point *= scl
        # len_new_guide = np.linalg.norm(vec_to_guide_point)
        # print(f"Old length {len_to_guide} wanted {len_to_wire} now {len_new_guide}")

        # Vec to guide point is now length of vec to wire
        vec_to_guide_point = vec_to_guide_point / len_to_wire
        vec_to_wire_point = vec_to_wire_point / len_to_wire
        # print(f"Vec to guide point {vec_to_guide_point}, vec to wire {vec_to_wire_point}")
        dot = np.dot(vec_to_guide_point, vec_to_wire_point)
        # If already pointing in the same direction, don't rotate
        if np.isclose(dot, 1.0):
            return scl, vec_to_guide_point, 0.0

        ang = np.acos(dot)
        # Cross product - spin around this vector
        vec_cross = np.cross(vec_to_guide_point, vec_to_wire_point)
        vec_cross = vec_cross / np.linalg.norm(vec_cross)
        return scl, vec_cross, ang

    def bend_to_wire(self, pt_origin: Vector3, heading: Vector3, left: Vector3):
        """
        Convert points to global coords, then do a pivot around each point in turn to incrementally align
        the nth point with the wire guide point
        Note: need to re-do if the initial point/vector change
        """
        rot_mat, gp_as_np = self._convert_guide_points_to_global(pt_origin=pt_origin, heading=heading, left=left)
        # print("Tie down control points in local coordinate system")
        #for indx in range(0, self.n_pts_tie_down+1):
        #    print(f"pre-move local {self.guide_points[indx]} global {gp_as_np[indx, 0:3]}")
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
            # print(f"Scale {scl} vec {vec} ang {ang}")
            # print(f"Tie point start {start_pt} wire {next_wire_pt} guide_point {next_guide_pt}")
            # print(f"Scl {scl} Vec {vec}, ang {ang}")
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
            # new_next_guide = gp_as_np[start_indx + n_spacing, 0:3]
            # vec_guide = new_next_guide - start_pt
            # vec_guide = vec_guide / np.linalg.norm(vec_guide)
            # vec_wire = next_wire_pt - start_pt
            # vec_wire = vec_wire / np.linalg.norm(vec_wire)
            # print(f"  Vector dot {np.dot(vec_guide, vec_wire)} New guide point loc {gp_as_np[start_indx + n_spacing, 0:3]}")
            start_indx += n_spacing
            n_spacing = self.n_pts_per_tie
            indx_guide_pts.append(start_indx)
        # print("first eight tied down")
        # for indx in range(0, self.n_pts_tie_down+1):
        #     print(f"pre-move local {self.guide_points[indx]} moved {gp_as_np[indx, 0:3]}")
        self.guide_points = []
        for indx in range(0, gp_as_np.shape[0]):
            # Put back at the branch location, but use global orientation
            # LPy will add a @R to reset orientation to the default (see base_lpy.lpy I module)
            pt = gp_as_np[indx, 0:3] - np.array(pt_origin)
            self.guide_points.append(Vector4(pt[0], pt[1], pt[2], 1.0))
        print(f"Curve start point: {pt_origin}\nFirst wire point: {self.wire_attach.attractor_pts[0]}")

        """
        print(f"Guide points to wire point alignment")
        np.set_printoptions(precision=2, suppress=True)
        for indx in range(0, self.wire_attach.attractor_pts.shape[0]):
            indx_guide = indx_guide_pts[indx+1]
            print(f"gp world {gp_as_np[indx_guide, 0:3]} wire {self.wire_attach.attractor_pts[indx]} gp local {gp_as_np[indx_guide, 0:3] - pt_origin} {self.guide_points[indx_guide]}")
        print("first eight")
        for indx in range(0, self.n_pts_tie_down+1):
            print(f"global {gp_as_np[indx, 0:3]}, local {self.guide_points[indx]}")
        """
        print("done")

    @staticmethod
    def _deflection_at_x(d, x, L):
        """d is the max deflection, x is the current location we need deflection on and L is the total length"""
        return (d / 2) * (x**2) / (L**3) * (3 * L - x)
