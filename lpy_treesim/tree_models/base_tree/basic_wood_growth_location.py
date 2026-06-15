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
    max_length: float = 1.0

    # These will be set to values based on vigour and default
    mean_length_growth_per_year: list[float] = None
    diameter: float = 0.001
    mean_diameter_growth_per_iteration: float = 0.1
    target_diameter_ratio = 0.025 / 1.0   # Ideal

    # For tracking growth; year will increment when a year completes
    year: int = 0   # Increments after every pruning cycle
    age: int = 0    # In iterations

    def __post_init__(self):
        if self.num_iter_per_year == -1:
            raise ValueError("Growth state: Forgot to set num_iter_per_year")
        if self.mean_length_growth_per_year is None:
            #  Really should never get here... but between 0.5 and 1.5 meters
            self.mean_length_growth_per_year = [(1.0 - self.vigour_level) * 0.5 + self.vigour_level * 1.5]
        # Sets how long to follow guide curve
        self.max_length = 0.0
        for l in self.mean_length_growth_per_year:
            self.max_length += 1.5 * l
        self.set_diameter_target()

    def set_diameter_target(self):
        """Calculate the target diameter based on the vigor level and expected growth for the year and current length
           Only call on year boundaries """
        estimated_length = self.length + self.length_growth_per_year()
        low_ratio = 0.01 / 1.0
        ideal_ratio = 0.025 / 1.0
        high_ratio = 3.5 / 1.0
        if self.vigour_level < 0.5:
            t = self.vigour_level * 2.0
            ratio = (1 - t) * low_ratio + t * ideal_ratio
        else:
            t = (self.vigour_level - 0.5) * 2.0
            ratio = (1 - t) * ideal_ratio + t * high_ratio
        if self.length > 0.0:
            cur_ratio = self.diameter / self.length_growth_per_year()
        else:
            cur_ratio = 0.0
        self.mean_diameter_growth_per_iteration = cur_ratio * (estimated_length) - self.diameter

    def length_growth_per_year(self):
        if self.year > len(self.mean_length_growth_per_year):
            return self.mean_length_growth_per_year[-1]
        return self.mean_length_growth_per_year[self.year]

    def mean_length_growth_per_iteration(self):
        return self.length_growth_per_year() / self.num_iter_per_year

    def get_diameter_increment(self):
        diameter_incr = self.lpy_rng.normal(loc=self.mean_diameter_growth_per_iteration, scale=0.1 * self.mean_diameter_growth_per_iteration)
        return diameter_incr

    def get_length_increment(self):
        # Generate a growth per iteration
        mean_length_growth = self.mean_length_growth_per_iteration()
        length_incr = self.lpy_rng.normal(loc=mean_length_growth, scale=0.1 * mean_length_growth)
        return length_incr
