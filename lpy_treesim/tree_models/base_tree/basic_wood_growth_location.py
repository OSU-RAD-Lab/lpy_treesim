"""
Defines the growth and location classes used in BasicWood

LocationState tracks start and end locations (set in base_lpy.lpy during the interpretation step, see IStartBranch,
   IEndBranch, GetPos, GetHead, GetLeft calls
GrowthState tracks the length of the branch and how much to grow each year. Each branch has an overall vigor level
   that controls overall growth. Growth ranges are set in the branch/spur config files (see BasicWoodConfig).
"""

from openalea.plantgl.all import Vector3
import numpy as np
from dataclasses import dataclass


@dataclass
class LocationState:
    """Location tracking for a wood object: start point & direction, end point and direction
    These values are filled in from the string during the interpretation stage"""

    # See LPy's turtle frame (heading, left, up)
    start: Vector3 = None
    start_dir: Vector3 = None
    start_left_dir: Vector3 = None
    end: Vector3 = None
    end_dir: Vector3 = None
    end_left_dir: Vector3 = None

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

    # Set this
    vigour_level: float = 0.5  # Between 0 and 1

    # Random number to use - this is here for repeatability
    #  These both come from the Branch configuration file for the specific tree type
    lpy_rng: np.random.Generator = None
    num_iter_per_year: int = -1

    # See config file for prune_length
    length: float = 0.0
    length_without_pruning: float = 0.0  # For diameter calculation - how much the branch grew altogether

    # These will be set to values based on vigour and default
    #   Note: Only need to set diameter at the beginning of the branch; the tapering will be handled
    #   in the cylinder generation.
    mean_length_growth_per_year: list[float] = None
    diameter: float = 0.001
    target_diameter_ratio = 0.025 / 1.0   # Ideal ratio of diameter to length
    taper: float = 0.1                    # Percentage of diameter to taper to. Don't make too small (< 0.01)
    target_ratios: tuple[float, float, float] = (0.01, 0.025, 0.035)

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
            # Make sure it's grown a bit at creation
            self.length = self.mean_length_growth_per_iteration()
            self.length_without_pruning = self.length

        # Based on vigor level. Diameter ratio is used in the growth step to set the starting thickness of the branch
        self._set_diameter_ratio()

    def _set_diameter_ratio(self):
        """Calculate the target diameter based on the vigor level and expected growth for the age_in_years and current length
           Called in init() method """
        low_ratio = self.target_ratios[0]
        ideal_ratio = self.target_ratios[1]
        high_ratio = self.target_ratios[2]
        if self.vigour_level < 0.5:
            t = self.vigour_level * 2.0
            self.target_diameter_ratio = (1 - t) * low_ratio + t * ideal_ratio
        else:
            t = (self.vigour_level - 0.5) * 2.0
            self.target_diameter_ratio = (1 - t) * ideal_ratio + t * high_ratio
        assert self.target_diameter_ratio > 0.0

    def length_growth_per_year(self):
        """ How much the branch should grow in one year, based on its current age"""
        if self.age_in_years >= len(self.mean_length_growth_per_year):
            return self.mean_length_growth_per_year[-1]
        return self.mean_length_growth_per_year[self.age_in_years]

    def mean_length_growth_per_iteration(self):
        return self.length_growth_per_year() / self.num_iter_per_year

    def get_length_increment(self):
        # Generate a growth per iteration
        mean_length_growth = self.mean_length_growth_per_iteration()
        if mean_length_growth < 0.0:
            print("oops")
        length_incr = self.lpy_rng.normal(loc=mean_length_growth, scale=0.1 * mean_length_growth)
        if length_incr < 0.00001:
            length_incr = 0.00001
        return length_incr

    def get_start_diameter(self):
        """ The diameter, based on the ratio """
        return self.target_diameter_ratio * self.length_without_pruning

    def get_end_diameter(self):
        """ The diameter, based on the ratio """
        return self.target_diameter_ratio * self.length_without_pruning * self.taper

    def get_diameter(self, dist_along: float = 0.5):
        """ Linear scale at the moment
        Switching to Vinci/Murray pipe model which is Diameter of parent is
        d^beta = sum all children d^ beta, with beta == 2 ish (1.5 resists bending, 3.0 maximizes flow)
        Exponential decay is more natural d base * (dtop / dbase) ^t"""
        t = dist_along / self.length_without_pruning
        if t < 0.0 or t > 1.0:
            print(f"Warning: Bad t value {t} in get_diameter")
        return (1.0 - t) * self.get_start_diameter() + t * self.get_end_diameter()
