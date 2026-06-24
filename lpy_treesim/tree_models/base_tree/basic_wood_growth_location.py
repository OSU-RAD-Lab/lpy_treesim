"""
Defines the abstract class BasicWood along with helper classes LocationState, GrowthState, InfoState and TyingState
"""

from openalea.plantgl.all import Vector3
import numpy as np
from dataclasses import dataclass


@dataclass
class LocationState:
    """Location tracking for a wood object: start point & direction, end point and direction
    These values are filled in from the string during the interpretation stage"""

    start: Vector3 = None      # Vector3
    start_dir: Vector3 = None  # Vector3
    start_left_dir: Vector3 = None  # Vector3
    end: Vector3 = None        # Vector3
    end_dir: Vector3 = None    # Vector3
    end_left_dir: Vector3 = None # Vector3

    def __post_init__(self):
        """Initialize Vector3 points if not provided."""
        if self.start is None:
            self.start = Vector3(0, 0, 0)
        if self.start_dir is None:
            self.start_dir = Vector3(1.0, 0, 0)
        if self.start_left_dir is None:
            self.start_left_dir = Vector3(1.0, 0, 0)
        if self.end is None:
            self.end = Vector3(0, 0, 0)
        if self.end_dir is None:
            self.end_dir = Vector3(0, 0, 0)
        if self.end_left_dir is None:
            self.end_left_dir = Vector3(0, 0, 0)


@dataclass
class GrowthState:
    """Growth parameters for a wood object. These will be set when constructed but with
       some noise, depending on the overall vigor level assigned to the branch"""

    # Set these
    vigour_level: float = 0.5  # Between 0 and 1

    # Random number to use - this is here for repeatability
    #  These both come from the Branch configuration file for the specific tree type
    lpy_rng: np.random.Generator = None
    num_iter_per_year: int = -1

    length: float = 0.0
    length_without_pruning: float = 0.0  # For diameter calculation - how much the branch grew altogether
    prune_length: float = 1.0            # Prune back to this length if need be

    # These will be set to values based on vigour and default
    #   Note: Only need to set diameter at the beginning of the branch; the tapering will be handled
    #   in the cylinder generation.
    mean_length_growth_per_year: list[float] = None
    diameter: float = 0.001
    target_diameter_ratio = 0.025 / 1.0   # Ideal
    taper: float = 0.1                    # Percentage of diameter to taper to

    # For tracking growth; controls which length/bud spacing to use
    age_in_years: int = 0         # Incremented by one after end of growth/tie/prune cycle
    age_in_iterations: int = 0    # In iterations

    def __post_init__(self):
        if self.num_iter_per_year == -1:
            raise ValueError("Growth state: Forgot to set num_iter_per_year")
        if self.mean_length_growth_per_year is None:
            #  Really should never get here... but between 0.5 and 1.5 meters
            print(f"Warning: Growth, did not set mean length growth per year")
            self.mean_length_growth_per_year = [(1.0 - self.vigour_level) * 0.5 + self.vigour_level * 1.5]

        for mlg in self.mean_length_growth_per_year:
            if mlg < 0.0:
                print(f"Error: mean langth growth per year negative {mlg}")
        if self.length <= 0.0:
            # Make sure it's grown a bit
            self.length = self.mean_length_growth_per_iteration()
            self.length_without_pruning = self.length

        # Based on vigor level. Diameter ratio is used in the growth step to set the starting thickness of the branch
        self.set_diameter_ratio()

    def set_diameter_ratio(self):
        """Calculate the target diameter based on the vigor level and expected growth for the age_in_years and current length
           Only call on age_in_years boundaries """
        low_ratio = 0.01
        ideal_ratio = 0.025
        high_ratio = 0.035
        if self.vigour_level < 0.5:
            t = self.vigour_level * 2.0
            self.target_diameter_ratio = (1 - t) * low_ratio + t * ideal_ratio
        else:
            t = (self.vigour_level - 0.5) * 2.0
            self.target_diameter_ratio = (1 - t) * ideal_ratio + t * high_ratio
        assert self.target_diameter_ratio > 0.0

    def length_growth_per_year(self):
        """ How much the branch should grow in one year, based on it's current age"""
        if self.age_in_years >= len(self.mean_length_growth_per_year):
            return self.mean_length_growth_per_year[-1]
        return self.mean_length_growth_per_year[self.age_in_years]

    def mean_length_growth_per_iteration(self):
        return self.length_growth_per_year() / self.num_iter_per_year

    def get_start_diameter(self):
        """ The diameter, based on the ratio """
        return self.target_diameter_ratio * self.length_without_pruning

    def get_end_diameter(self):
        """ The diameter, based on the ratio """
        return self.target_diameter_ratio * self.length_without_pruning * self.taper

    def get_diameter(self, t: float = 0.5):
        """ Linear scale at the moment"""
        return (1.0 - t) * self.get_start_diameter() + t * self.get_end_diameter()

    def get_length_increment(self):
        # Generate a growth per iteration
        mean_length_growth = self.mean_length_growth_per_iteration()
        if mean_length_growth < 0.0:
            print("oops")
        length_incr = self.lpy_rng.normal(loc=mean_length_growth, scale=0.1 * mean_length_growth)
        if length_incr < 0.00001:
            length_incr = 0.00001
        return length_incr
