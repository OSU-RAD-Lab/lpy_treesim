from dataclasses import dataclass
from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig, TreeSimulationBase
from lpy_treesim.tie_prune.wire_support import Support, TyingState


@dataclass
class ENVYSimulationConfig(SimulationConfig):
    """Configuration for Envy trellis tree simulation parameters."""

    # Envy-specific Support Structure
    start_height: float = 0.6   # 24 inches
    angle: float = 0.0
    spacing_wires: float = 0.45 # 18 inches
    num_wires: int = 4          # Make a bit taller than 5
    x_left: float = -0.7      # changed from 0.6 (2 feet) on either side
    x_right: float = 0.7
    n_years: int = 6
    n_along: int = 4 # changed from 3 tied down points


class ENVYSimulation(TreeSimulationBase):
    """
    Envy trellis architecture simulation.

    Implements the Envy V-trellis training system with wires arranged in a V-shape
    on both sides of the tree row.
    """

    def generate_attractor_grids(self, config: SimulationConfig):
        """
        Generate 3D points for the Envy trellis wire structure.

        Trunk: Trunk is tied along the wires
        Branches: first level support branches are tied along the wires at evenly spaced points

        Returns:
            The support class
        """
        # The trunk support
        self.trunk_attractor = self.support.make_attractor_grid(tie_type=TyingState.TyingType.TIE_ACROSS, n_along_x=1)
        self.branch_attractor = self.support.make_attractor_grid(tie_type=TyingState.TyingType.TIE_ALONG, n_along_x=config.n_along) # Changed from 3

        return self.support
