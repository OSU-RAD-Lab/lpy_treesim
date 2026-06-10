from dataclasses import dataclass
from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig, TreeSimulationBase
from lpy_treesim.tie_prune.wire_support import Support, TyingState


@dataclass
class UFOSimulationConfig(SimulationConfig):
    """Configuration for UFO trellis tree simulation parameters."""

    # Override base defaults for UFO-specific values
    num_iteration_tie: int = 8
    num_iteration_prune: int = 16
    pruning_age_threshold: int = 8
    derivation_length: int = 160

    # UFO-specific Support Structure
    start_height: float = 0.5
    angle: float = 0.0
    spacing_wires: float = 0.45
    num_wires: int = 6
    x_left: float = 0.0   # Start at the trunk center
    x_right: float = 2.0

    # UFO-specific Point Generation
    ufo_x_range: tuple = (0.65, 3)
    ufo_x_spacing: float = 0.3
    ufo_z_value: float = 1.4
    ufo_y_value: float = 0

    # UFO-specific Growth Parameters
    thickness_multiplier: float = 1.2  # Multiplier for internode thickness

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
