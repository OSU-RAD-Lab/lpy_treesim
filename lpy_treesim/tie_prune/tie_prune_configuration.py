from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import truediv

import numpy as np


@dataclass
class SimulationConfig(ABC):
    """Configuration parameters for the actual simulation part

    Architecture-specific configs should inherit from this and add their own parameters.
    Common parameters across all architectures are defined here.

    Contains information on:
    - The support structure
    - Default values for how many iterations equals one age_in_years's worth of growth
    - Parameters for determining amount of effort needed to tie a branch to a wire
    """

    # Tying and Pruning Intervals - pruning will happen after tying.
    #  If prune_summer is true, then pruning will also happen at 3/4 point
    num_iter_per_year: int = 28
    prune_summer: bool = False

    # Energy Parameters
    energy_distance_weight: float = 0.5  # Weight for distance in energy calculation
    energy_threshold: float = 1.0  # Maximum energy threshold for tying

    # Support parameters - override these to get wires at different heights
    # start height - first wire
    # angle - optional tilt the entire structure
    # x_left/right are relative to the tree trunk, which is at 0
    start_height: float = 0.5
    angle: float = 0.0
    spacing_wires: float = 0.45
    num_wires: int = 6
    x_left: float = -1.0
    x_right: float = 1.0

    # Pruning Parameters
    pruning_age_threshold: int = 6  # Age threshold for pruning untied branches

    # L-System Parameters
    n_years:int = 6  # for a total of num_iter_per_year * n_year derivation steps

    # Growth Parameters
    tolerance: float = 1e-5  # Tolerance for comparison between floats

    cylinder_length: float = 0.03                # Generate cylinders every inch or so

    # Visualization Parameters
    attractor_point_width: int = 10  # Width of attractor points in visualization

    # Seed for randomization
    seed: int = np.random.uniform(-1200, 1200)

    # Random number to use - this is here for repeatability
    lpy_rng: np.random.Generator = None

    def __post_init__(self):
        self.lpy_rng = np.random.default_rng(self.seed)

    def end_height(self):
        return self.start_height + self.num_wires * self.spacing_wires

    @property
    def derivation_length(self):
        return self.n_years * self.num_iter_per_year

    def do_trunk_tying(self, current_iteration: int):
        # Tie the iteration before branch tying so shape propagates correctly
        return False
        if current_iteration % (self.num_iter_per_year - 1) == 0:
            return True
        return False

    def do_branch_tying(self, current_iteration: int):
        # Tie the iteration before branch tying so shape propagates correctly
        if current_iteration % (self.num_iter_per_year) == 0:
            return True
        return False

    def do_pruning(self, current_iteration: int):
        # Prune the iteration after tying (and at 3/4 of growth if doing summer pruning)
        if current_iteration % (self.num_iter_per_year + 1) == 0:
            return True
        if self.prune_summer:
            prune_iteration = 3 * self.num_iter_per_year // 4
            if current_iteration % prune_iteration == 0:
                return True
        return False

    def do_year_increment(self, current_iteration: int):
        if current_iteration % self.num_iter_per_year == 1:
            if current_iteration > 1:
                return True
        return False
