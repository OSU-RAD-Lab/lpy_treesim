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

    start: any = None      # Vector3
    start_dir: any = None  # Vector3
    end: any = None        # Vector3
    end_dir: any = None    # Vector3
    end_left_dir: any = None # Vector3

    def __post_init__(self):
        """Initialize Vector3 points if not provided."""
        if self.start is None:
            self.start = Vector3(0, 0, 0)
        if self.start_dir is None:
            self.start_dir = Vector3(1.0, 0, 0)
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

        # Based on vigor level. Diameter ratio is used in the growth step to set the starting thickness of the branch
        self.set_diameter_ratio()

    def set_diameter_ratio(self):
        """Calculate the target diameter based on the vigor level and expected growth for the age_in_years and current length
           Only call on age_in_years boundaries """
        low_ratio = 0.01 / 1.0
        ideal_ratio = 0.025 / 1.0
        high_ratio = 3.5 / 1.0
        if self.vigour_level < 0.5:
            t = self.vigour_level * 2.0
            self.target_diameter_ratio = (1 - t) * low_ratio + t * ideal_ratio
        else:
            t = (self.vigour_level - 0.5) * 2.0
            self.target_diameter_ratio = (1 - t) * ideal_ratio + t * high_ratio

    def length_growth_per_year(self):
        """ How much the branch should grow in one year, based on it's current age"""
        if self.age_in_years > len(self.mean_length_growth_per_year):
            return self.mean_length_growth_per_year[-1]
        return self.mean_length_growth_per_year[self.age_in_years]

    def mean_length_growth_per_iteration(self):
        return self.length_growth_per_year() / self.num_iter_per_year

    def get_start_diameter(self):
        """ The diameter, based on the ratio """
        return self.target_diameter_ratio * self.prune_length

    def get_end_diameter(self):
        """ The diameter, based on the ratio """
        end_diameter = self.target_diameter_ratio * self.taper
        # Now account for cutting off the end
        if self.length < self.prune_length:
            end_diameter *= self.length / self.prune_length
        return end_diameter

    def get_length_increment(self):
        # Generate a growth per iteration
        mean_length_growth = self.mean_length_growth_per_iteration()
        length_incr = self.lpy_rng.normal(loc=mean_length_growth, scale=0.1 * mean_length_growth)
        return length_incr
