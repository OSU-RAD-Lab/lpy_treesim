"""
Defines the abstract class BasicWoodConfig which controls how the tree grows
"""
import numpy as np
from dataclasses import dataclass

from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tree_models.base_tree.bud_site import BudSite


@dataclass
class BasicWoodConfig:
    """ Configuration parameters for BasicWood initialization.
    During one LPy interation the branch will grow some amount along a guide curve (if tied) or along
      a starting direction, curving up.
    The intent is to mimic the tie/prune cycle, so growth parameters are given in terms of yearly growth
    Given a fixed number of iterations per age_in_years, we can determine the rest of the parameters
    Buds for cherries are (usually) clearly vegetative versus fruiting, apples are a mix

    For cherrys/apples (most fruit trees) growth in length and angle and bud placement can be characterized
      as follows:
      - Amount a branch will grow in a given age_in_years (usually more in the first 1-3 years)
      - Bud spacing along the branch (how far apart are buds, on average?)
      - Bud angle wrt the parent branch - this can vary by bud type (vegetative or fruit) and
      - Bud angle on the branch - spiral pattern, 2/5 phyllotaxis, ie, the next bud will be a rotation of
         144 degrees from the last one
      - Branch diameter vs length is usually related, and follows
         Ideal: < 2.5cm diameter per meter of length
         Vigorous: approaching 3cm / m of length
    [Note - these assume a dwarfing root stock of some sort]

    Given parameters:
      - number of iterations that equal one age_in_years
      - A list of yearly growth length ranges by age_in_iterations eg, 12-36" in first age_in_years, 12-24" in 3rd age_in_years
      - Average bud spacing
      - Phyllotaxis angle
      - Bud angle range by type [vegetative versus fruiting, versus mixed]
      - Overall vigor - what percentage of the branches should be given 'vigorous' growth values
      - Overall curviness of branches (used to create guide curves)
      - What type of tying to do (if any)
      - If the branch is prunable

      - Note: Stopping (or slowing growth) can be handled by having the last yearly_growth_range be really small
    Derived parameters
      - How much to grow by at each iteration
      - Diameters (initial) and diameter growth rates

    Notes on the LString:
    Within the lpy string creation, specifically grow_object, whenever a new bud is started (should_bud
       returned true) then lpy inserts a parameter with the branch and the number of buds (set to zero)
       Should bud defines a segment length to be total"""

    # Growth variables
    bud_spacing_range: tuple = (0.0254, 0.0508)  # 1-2 inches
    yearly_growth_range: list[tuple] = None      # eg (1, 24, 36) would be up to 1 age_in_years between 24 and 36 inches
    taper_amount: float = 0.5                    # How much to taper by
    phyllotaxis_angle: float = 144               # How to space buds around a branch
    bud_angle_probs: dict = None                # Bud angle relative to branch; angle may depend on type of bud
    bud_break_probs: tuple = (0.1, 0.3, 0.35)   # EG , will turn vegetative with 0.1 prob, fruiting w 0.3 - 0.1

    # Curve parameters for L-System growth guides
    #   Since growth curves are always in the heading direction (0,0,1) wiggle in x and y but straight in z
    curve_x_range: tuple = (-0.25, 0.25)  # X noise bounds for Bezier curve control points
    curve_y_range: tuple = (-0.25, 0.25)  # Y noise bounds for Bezier curve control points

    # Tying and pruning variables
    tie_type: TyingState.TyingType = TyingState.TyingType.NO_TIE # How to tie this branch type to support
    tie_start_dist: float = 0.46                # (18 inches) Expected distance from the base of the trunk/branch to first tie point
    tie_spacing: float = 0.46                   # (18 inches) Spacing of tie points

    prune_length: float = 1000.0                 # If you want to have the branch pruned after a certain length...
    prunable: bool = True

    # Random number to use - this is here for repeatability
    lpy_rng: np.random.Generator = None
    num_iter_per_year: int = -1

    # Store the prototype configs here
    configs_dict: dict = None,

    def __post_init__(self):
        """Validate geometric parameters for consistent growth behavior."""
        if self.num_iter_per_year == -1:
            raise ValueError("BasicWoodConfig: Forgot to initialize num_iter_per_year")
        if self.yearly_growth_range is None:
            """ Set to 24-36 inches the first 3 years, 12-24 for the next 3, then 2-6"""
            self.yearly_growth_range = []
            for _ in range(0, 3):
                self.yearly_growth_range.append((0.3, 0.6))
            self.yearly_growth_range.append((0.05, 0.1))
        if self.bud_angle_probs is None:
            self.bud_angle_probs = {BudSite.BudType.VEGETATIVE: (15, 35),
                                    BudSite.BudType.FRUITING: (30, 50),
                                    BudSite.BudType.MIXED: (15, 50)}
