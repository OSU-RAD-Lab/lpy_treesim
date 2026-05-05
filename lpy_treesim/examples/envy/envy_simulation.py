from lpy_treesim.simulation_base import SimulationConfig, TreeSimulationBase
from dataclasses import dataclass
import numpy as np


@dataclass
class ENVYSimulationConfig(SimulationConfig):
    """Configuration for Envy trellis tree simulation parameters."""

    # Override base defaults for Envy-specific values
    num_iteration_tie: int = 5
    num_iteration_prune: int = 8
    pruning_age_threshold: int = 6
    derivation_length: int = 128

    # Envy-specific Support Structure
    support_trunk_wire_point = None
    support_num_wires: int = 14

    # Envy-specific Point Generation (V-trellis)
    trellis_x_value: float = 0.45
    trellis_z_start: float = 0.6
    trellis_z_end: float = 3.4
    trellis_z_spacing: float = 0.45

    use_generalized_cylinders: bool = False


class ENVYSimulation(TreeSimulationBase):
    """
    Envy trellis architecture simulation.

    Implements the Envy V-trellis training system with wires arranged in a V-shape
    on both sides of the tree row.
    """

    def generate_points(self):
        """
        Generate 3D points for the V-trellis wire structure.

        Creates a linear array of wire attachment points along the x-axis at a fixed
        height (z) and depth (y). The points are spaced evenly within the configured
        z-range and used to construct the trellis support structure.

        Returns:
            list: List of (x, y, z) tuples representing wire attachment points in V-trellis formation
        """
        x = np.full((7,), self.config.trellis_x_value).astype(float)
        y = np.full((7,), 0).astype(float)
        z = np.arange(self.config.trellis_z_start, self.config.trellis_z_end, self.config.trellis_z_spacing)

        pts = []
        for i in range(x.shape[0]):
            pts.append((-x[i], y[i], z[i]))
            pts.append((x[i], y[i], z[i]))
        return pts
