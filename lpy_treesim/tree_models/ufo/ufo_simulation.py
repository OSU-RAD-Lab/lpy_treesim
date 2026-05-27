from dataclasses import dataclass
import numpy as np
from lpy_treesim.simulation_base import SimulationConfig, TreeSimulationBase


@dataclass
class UFOSimulationConfig(SimulationConfig):
    """Configuration for UFO trellis tree simulation parameters."""

    # Override base defaults for UFO-specific values
    num_iteration_tie: int = 8
    num_iteration_prune: int = 16
    pruning_age_threshold: int = 8
    derivation_length: int = 160

    # UFO-specific Support Structure
    support_trunk_wire_point: tuple = (0.6, 0, 0.4)
    support_num_wires: int = 7

    # UFO-specific Point Generation
    ufo_x_range: tuple = (0.65, 3)
    ufo_x_spacing: float = 0.3
    ufo_z_value: float = 1.4
    ufo_y_value: float = 0

    # UFO-specific Growth Parameters
    thickness_multiplier: float = 1.2  # Multiplier for internode thickness

    use_generalized_cylinders: bool = False


class UFOSimulation(TreeSimulationBase):
    """
    UFO trellis architecture simulation.

    Implements the UFO (Upright Fruiting Offshoots) training system with
    horizontal wires arranged linearly along the x-axis.
    """

    def generate_points(self):
        """
        Generate 3D points for the UFO trellis wire structure.

        Creates a linear array of wire attachment points along the x-axis at a fixed
        height (z) and depth (y). The points are spaced evenly within the configured
        x-range and used to construct the trellis support structure.

        Returns:
            list: List of (x, y, z) tuples representing wire attachment points,
                  where all points share the same y and z coordinates.
        """
        x = np.arange(self.config.ufo_x_range[0], self.config.ufo_x_range[1], self.config.ufo_x_spacing).astype(float)
        z = np.full((x.shape[0],), self.config.ufo_z_value).astype(float)
        y = np.full((x.shape[0],), self.config.ufo_y_value).astype(float)

        wire_attachment_points = []
        for point_index in range(x.shape[0]):
            wire_attachment_points.append((x[point_index], y[point_index], z[point_index]))

        return wire_attachment_points
