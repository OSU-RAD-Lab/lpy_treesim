from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig, TreeSimulationBase
from dataclasses import dataclass
from lpy_treesim.tie_prune.wire_support import Support, TyingState


@dataclass
class ENVYSimulationConfig(SimulationConfig):
    """Configuration for Envy trellis tree simulation parameters."""

    # Override base defaults for Envy-specific values
    num_iteration_tie: int = 5
    num_iteration_prune: int = 8
    pruning_age_threshold: int = 6
    derivation_length: int = 64

    # Envy-specific Support Structure
    support_trunk_wire_point = None
    support_num_wires: int = 14

    # Envy-specific Point Generation (V-trellis)
    trellis_x_value: float = 0.45
    trellis_z_start: float = 0.6
    trellis_z_end: float = 3.4
    trellis_z_spacing: float = 0.45

    use_generalized_cylinders: bool = True


class ENVYSimulation(TreeSimulationBase):
    """
    Envy trellis architecture simulation.

    Implements the Envy V-trellis training system with wires arranged in a V-shape
    on both sides of the tree row.
    """

    def generate_attractor_grids(self):
        """
        Generate 3D points for the UFO trellis wire structure.

        Trunk: Trunk is tied along the wires
        Branches: first level support branches are tied along the wires at evenly spaced points

        Returns:
            The support class
        """
        # The trunk support
        self.trunk_attractor = self.support.make_attractor_grid(tie_type=TyingState.TyingType.TIE_ACROSS, n_along_x=1)
        self.branch_attractor = self.support.make_attractor_grid(tie_type=TyingState.TyingType.TIE_ALONG, n_along_x=3)

        return self.support
