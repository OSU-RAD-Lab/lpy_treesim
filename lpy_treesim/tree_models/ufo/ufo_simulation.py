from dataclasses import dataclass
from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig, TreeSimulationBase
from lpy_treesim.tie_prune.wire_support import Support, TyingState
import numpy as np
from openalea.plantgl.all import BezierCurve, Vector4, Point4Array


@dataclass
class UFOSimulationConfig(SimulationConfig):
    """Configuration for UFO trellis tree simulation parameters.
    Based on: https://www.cloudmountainfarmcenter.org/upright-fruiting-offshoot-ufo-sweet-cherry-training/
    Spacing: 5-6' (1.5-1.82 meters)
    5 wires, lowest at 20", spaced 18-20 inches (20 inches = 0.5m)
    Start growing at a 45 degree angle
    Remove all buds below first trellis wire
    train at 12 inches (0.3m)
    Space 8-10 inches apart and remove shoots below horizontal (0.2-0.254m)

    Buds are spaced 1-2 inches apart on dwarfing root stock (0.0254m - 0.0508m)
    Branches grow 12-24 inches in one season (mature), 24-36 inches (first 3 years)
      -- (0.305m - 0.61m) and (0.61m - 0.94m)
    Target: 12-24 inches per age_in_years for best cherry production

    For branches under three years they should grow 24-36 inches and have buds every 1-2 inches.
    From an lpy iteration standpoint, that's 24/2 = 12 to 36/1 = 36 growth steps (call it 24)
    So iterations between tying should be around 24/1.7 = 14 (assuming mean bud spacing of 1.7)
    """

    # UFO-specific Support Structure
    start_height: float = 0.5   # 20 inches
    angle: float = 0.0
    spacing_wires: float = 0.45 # 18 inches
    num_wires: int = 6          # Make a bit taller than 5
    x_left: float = 0.0         # Start at the trunk center
    x_right: float = 1.8        # 5-6 ' to the right
    n_years:int = 4

    # UFO-specific Point Generation
    ufo_x_range: tuple = (0.65, 3)
    ufo_x_spacing: float = 0.3
    ufo_z_value: float = 1.4
    ufo_y_value: float = 0

    use_generalized_cylinders: bool = True


class UFOSimulation(TreeSimulationBase):
    """
    UFO trellis architecture simulation.

    Implements the UFO (Upright Fruiting Offshoots) training system with
    horizontal wires arranged linearly along the x-axis.
    """

    def generate_attractor_grids(self):
        """
        Generate 3D points for the UFO trellis wire structure.

        Trunk: Trunk is tied along the bottom-most wire but always with an upward trend
        Branches: Wire attachment points are along the x-axis at evenly-spaced intervals.

        Returns:
            The support class
        """
        # The trunk support
        #   Pin at 2 points along first wire
        self.trunk_attractor = self.support.make_atractor_grid(tie_type=TyingState.TyingType.TIE_ALONG_FIRST, n_along_x=4)
        self.branch_attractor = self.support.make_atractor_grid(tie_type=TyingState.TyingType.TIE_ACROSS, n_along_x=6)

        return self.support

    def create_trunk_curve(self):
        """ Create an initial growth curve that is at a 45 degree angle (which is how trees are planted)"""
        control_pts = []
        n_pts = 6
        angle = np.pi / 4.0
        # Assume one age_in_years before first tie, should grow 24-36 inches, so make the guide curve at least 3 feet long (1 m)
        curve_length = 1.0
        delta_step = curve_length / n_pts
        for n in range(0, n_pts):
            dt = (n+1) * delta_step
            x = dt * np.sin(angle) + self.config.lpy_rng.uniform(-0.1, 0.1)
            y = self.config.lpy_rng.uniform(-0.1, 0.1)
            z = dt * np.cos(angle)
            control_pts.append(Vector4(x, y, z, 1))
        control_point_array = Point4Array(control_pts)
        curve = BezierCurve(control_point_array)
        return curve
