"""
Defines the abstract classes for wires and supports
"""

import numpy as np
from dataclasses import dataclass
from lpy_treesim.tie_prune.tying import TyingState, WireBranchAttach
from openalea.plantgl.all import Point3Grid, Vector3


@dataclass
class Wire:
    # All wires are (currently) horizontal, only keep height and in-out amount (if wires are at an angle)
    #   Keep a unique id for each wire. The class WireBranchAttach will hold actual attachment points
    id: int = -1
    height: float = 0.0
    in_out: float = 0.0


class Support:
    """All the details needed to figure out how the support is structured in the environment,
       it is a collection of wires at set heights with a specific angle,
       x_left,right are the expected span of one tree, with tree base at 0,0,0 """

    def __init__(self,
                 start_height: float,
                 angle: float = 0.0,
                 spacing_wires: float = 0.46,
                 num_wires: int = 6,
                 x_left: float = -1.0,
                 x_right: float = 1.0,
                 n_along_x: int = 6):
        """ All lengths in meters, assumes base of post is along the 1, 0, 0 axis
        @ angle is in degrees, assuming tilting in the positive y direction"""

        self.spacing_wires = spacing_wires

        self.wires = []
        height = start_height
        ang_in_radians = (2.0 * np.pi) * angle / 360.0
        points = []
        x_left_right = np.linspace(x_left, x_right, n_along_x)
        for n in range(0, num_wires):
            in_out = height * np.sin(ang_in_radians)
            new_wire = Wire(id=n, height=height, in_out=in_out)
            self.wires.append(new_wire)

            for x in x_left_right:
                points.append((x, in_out, height))

            height += spacing_wires

        self.x_left = x_left
        self.x_right = x_right
        self.n_along_x = n_along_x

        # For visualization only - every time make_attractor_grid is called it will add its points here
        self.attractor_grids = []
        self.attractor_grid = Point3Grid((1, 1, 1), list(points))

    def spacing_across_wire(self):
        return (self.x_right - self.x_left) / self.n_along_x

    def make_attractor_grid(self,
                            tie_type: TyingState.TyingType = TyingState.TyingType.TIE_ALONG,
                            n_along_x: int = 1,
                            dx_single: float = 0.0,
                            start_x: float = -1.0,
                            skip_first: bool = False):
        """ Organize by branch/trunk, eg, branch[1] has to be tied to points 1,2,3,
            Each row is the pin points for one branch
            Can call multiple times, eg, once to get trunk once to get side branches"""

        tie_downs = []
        width = max(abs(self.x_left), abs(self.x_right))
        if start_x == -1.0:
            start_x = width / (n_along_x + 1.0)
        if tie_type == TyingState.TyingType.TIE_ALONG or tie_type == TyingState.TyingType.TIE_ALONG_FIRST:
            # Generate tie points along x, one for each wire (or only first wire)
            for wire in self.wires:
                x_start_stop = []
                if np.abs(self.x_left) > 0:
                    x_start_stop.append((-start_x, self.x_left))
                if np.abs(self.x_right) > 0:
                    x_start_stop.append((start_x, self.x_right))

                for (x_left, x_right) in x_start_stop:
                    dx = np.linspace(x_left, x_right, n_along_x)
                    tie_down = WireBranchAttach()
                    attractor_pts = []
                    for x in dx:
                        attractor_pts.append((x, wire.in_out, wire.height))
                    tie_down.attractor_pts = np.array(attractor_pts)
                    tie_downs.append(tie_down)
                if tie_type == TyingState.TyingType.TIE_ALONG_FIRST:
                    break
        elif tie_type == TyingState.TyingType.TIE_ACROSS:
            if n_along_x > 1:
                # Branches - don't use x = 0
                if self.x_left < 0.0 and self.x_right > 0.0:
                    dx = np.linspace(self.x_left, -start_x, n_along_x // 2)
                    np.hstack(np.linspace(start_x, self.x_right, n_along_x // 2))
                else:
                    if self.x_left < 0.0:
                        dx = np.linspace(self.x_left, -start_x, n_along_x)
                    else:
                        dx = np.linspace(start_x, self.x_right, n_along_x)
            else:
                # Trunk - the only one that should have tie downs along 0
                dx = [dx_single]
            for x in dx:
                tie_down = WireBranchAttach()
                attractor_pts = []
                for wire_indx, wire in enumerate(self.wires):
                    if wire_indx == 0 and skip_first:
                        continue
                    attractor_pts.append((x, wire.in_out, wire.height))
                tie_down.attractor_pts = np.array(attractor_pts)
                tie_downs.append(tie_down)

        # Set the direction and spacing for all tie downs
        for tie_down in tie_downs:
            tie_down.attractor_dir = tie_down.attractor_pts[1, :] - tie_down.attractor_pts[0, :]
            tie_down.spacing = float(np.linalg.norm(tie_down.attractor_dir))
            tie_down.attractor_dir /= tie_down.spacing

            # Add points to display
            for pt in tie_down.attractor_pts:
                pt_vec = Vector3(pt[0], pt[1], pt[2])
                self.attractor_grid.add_point(pt_vec)
        self.attractor_grids.append(tie_downs)
        return tie_downs
