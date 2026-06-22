"""
Defines the abstract class BasicWood along with helper classes LocationState, GrowthState, InfoState and TyingState
"""

from abc import ABC, abstractmethod
import copy
import numpy as np
from openalea.plantgl.scenegraph import BezierCurve
from openalea.plantgl.scenegraph.cspline import CSpline

from lpy_treesim.lpy_functions.lpy_geometry_fns import create_bezier_curve
from lpy_treesim.tree_models.base_tree.basic_wood_growth_location import GrowthState, LocationState
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tree_models.base_tree.basic_wood_config import BasicWoodConfig
import logging


class BasicWood(ABC):
    @staticmethod
    def clone(obj):
        try:
            return copy.deepcopy(obj)
        except copy.Error:
            raise copy.Error(f"Not able to copy {obj}") from None

    def __init__(self, config:BasicWoodConfig):
        # Unique name
        self.name: str = ""

        # Parameters controlling growth
        self.config = config
        if not isinstance(config, BasicWoodConfig):
            raise ValueError("Config should be tree-specific config inherited from BasicWoodConfig")

        # Track end points and orientations
        self.location = LocationState()

        # Tying status - contains what type of tying and attractor points, plus guide points
        self.tying = TyingState(tie_type=config.tie_type)

        # Track cut y/n
        self.cut = False

        # Growth Variables  - generate variables stochastically
        vigor = self.config.lpy_rng.normal(0.5, 0.2)
        if vigor < 0.0:
            vigor = 0.0
        if vigor > 1.0:
            vigor = 1.0

        # Use the vigor setting plus some noise to establish growth rate for this branch
        mgl = []
        start_year = 0
        for mgl_item in self.config.yearly_growth_range:
            for yr in range(start_year, mgl_item[0]):
                mean_range = (1.0 - vigor) * mgl_item[1] + vigor * mgl_item[2]
                mean_sd = 0.2 * (mgl_item[2] - mgl_item[1])
                mean_value = self.config.lpy_rng.normal(mean_range, mean_sd)
                if mean_value < mgl_item[1]:
                    mean_value = mgl_item[1]
                if mean_value > mgl_item[2]:
                    mean_value = mgl_item[2]
                mgl.append(mean_value)
                start_year += 1

        # Parameters controlling growth
        self.growth = GrowthState(vigour_level=vigor,
                                  mean_length_growth_per_year=mgl,
                                  prune_length=config.prune_length,
                                  num_iter_per_year=self.config.num_iter_per_year,
                                  taper=self.config.taper_amount,
                                  lpy_rng=self.config.lpy_rng)

        # Growth curve
        self.growth_curve: BezierCurve = self.initial_growth_curve()

        # Where all the budsites are located
        self.bud_angle_around = self.config.lpy_rng.uniform(0.0, 260.0)
        self.bud_sites = []
        self.logger = logging.getLogger(__name__)

    def __copy_constructor__(self, copy_from):
        update_dict = copy.deepcopy(copy_from.__dict__)
        for k, v in update_dict.items():
            setattr(self, k, v)
        # self.__dict__.update(update_dict)

    def dist_last_bud(self)->float:
        if len(self.bud_sites) == 0:
            return 0.0
        return self.bud_sites[-1].dist_along

    def growth_since_last_bud(self)->float:
        return self.growth.length - self.dist_last_bud()

    def is_add_bud_site(self) -> bool:
        """This method defines if a bud site should be added here. By default, generates a random number
        for the next bud site location relative to the last and if it's far enough away, generate one"""
        bud_split_mean = ((1.0 - self.growth.vigour_level) * self.config.bud_spacing_range[0] +
                          self.growth.vigour_level * self.config.bud_spacing_range[1])
        bud_split_sd = 0.2 * (self.config.bud_spacing_range[1] - self.config.bud_spacing_range[0])
        min_length_bud_site = self.config.lpy_rng.normal(bud_split_mean, bud_split_sd)
        if min_length_bud_site < self.config.bud_spacing_range[0]:
            min_length_bud_site = self.config.bud_spacing_range[0]
        if min_length_bud_site > self.config.bud_spacing_range[1]:
            min_length_bud_site = self.config.bud_spacing_range[1]

        len_from_last = self.growth_since_last_bud()
        if min_length_bud_site < len_from_last:
            return True
        return False

    def add_bud_site(self):
        """Add a dormant bud at the current site"""
        bud_name = self.name + f"_bud_{len(self.bud_sites)}"
        bud_angle_around = self.bud_angle_around
        self.bud_angle_around += self.config.phyllotaxis_angle + self.config.lpy_rng.uniform(-5, 5)
        while self.bud_angle_around > 360.0:
            self.bud_angle_around -= 360.0
        bud = BudSite(name=bud_name,
                      bud_break_probabilities=self.config.bud_break_probs,
                      bud_angle_probabilities=self.config.bud_angle_probs,
                      bud_angle_around=bud_angle_around,
                      wood_parent=self,
                      dist_along=self.growth.length,
                      lpy_rng=self.config.lpy_rng)
        self.bud_sites.append(bud)
        return bud

    def pre_bud_rule(self) -> list:
        """This method can define any internal changes happening to the properties of the class,
           such as reduction in thickness increment etc.
           Returns a list of lpy production rules (if any)"""
        return []

    def post_bud_rule(self) -> list:
        """This method can define any internal changes happening to the properties of the class,
        such as reduction in thickness increment etc.
        Returns a list of lpy production rules (if any)"""
        return []

    def grow(self):
        self.growth.age_in_iterations += 1
        length_incr = self.growth.get_length_increment()
        self.growth.length += length_incr
        self.growth.length_without_pruning += length_incr

    def prune_growth_length(self):
        """ Call during pruning step to prune back long growth"""
        if self.growth.length_without_pruning > self.growth.prune_length:
            # Add a bit of noise to model pruning cut noise
            target_length = self.config.lpy_rng.normal(self.growth.prune_length, 0.01)
            if target_length < 0.0001:
                target_length = 0.0001

            self.growth.length = target_length
            prune_buds = []
            for indx, bud in enumerate(self.bud_sites):
                # These *SHOULD* be in length order
                if bud.dist_along > target_length:
                    prune_buds = self.bud_sites[indx:]
                    del self.bud_sites[indx:]
                    return prune_buds
        return None

    def add_year(self):
        self.growth.age_in_years += 1

    @abstractmethod
    def create_branch(self):
        """ Returns how a new branch when bud break happens. Eg, if trunk, makes primary branch
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on it's parent when the bud breaks
        Should return an instance of a class that inherits fromBasic Brqnch"""
        pass

    @abstractmethod
    def create_spur(self):
        """ Creates a new spur/fruiting site.
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on it's parent when the bud breaks
        Should return an instance of a class that inherits from BasicSpur"""
        pass

    def initial_growth_curve(self):
        """ If not over-ridden later by tying, generate a growth curve.
        Note: This is relative to the starting direction of the wood object - so z is always 'out' """
        return create_bezier_curve(num_control_points=6,
                                   x_range=self.config.curve_x_range,
                                   y_range=self.config.curve_y_range,
                                   total_length=self.growth.prune_length)

    def contour_curve(self):
        # If you want a non-circular cross section... use lpy_geometry functions to make a contour
        return None

    def update_guide(self):
        """ If the branch/trunk has grown past the last tie point then append more guide points
        Also updates the tying variables (last tie point, guide points, guide_length)

        Notes:
            - If out of tie points sets guide_length to be zero (no longer follow curve)
            - Appends guide points incrementally to self.tying.guide_points
            - Only generates a new set of guide points when the branch crosses a tying point
            - Uses self.location.start as base if not yet tied; the last tie point in WireBranchAttach otherwise.
            - Adds some stochasticity to the guide curve by 1) letting across tie points slide in x and 2) following the
               direction of the branch growth (curve 'bows' out of guide)
        """
        if not self.tying.is_tied:
            return

        self.logger.info(f"Updating guide {self.name} {self.tying.last_tie_index}")
        # Ran out of tie points
        if self.tying.last_tie_index >= self.tying.wire_attach.attractor_pts.shape[1]:
            self.logger.info(f"  Off end")
            return

        # Case 1: We haven't started tying yet, so create a guide curve that goes from the end point
        #         to the first tie point
        # Case 2: We are still growing along the guide curve, haven't reached the end
        # Case 3: We have grown past the last tie point and need to add to the guide curve
        # Sets self.tying.guide_points and self.tying.guide_length
        self.tying.tie_needs_updating = False
        print(f"Branch {self.name}", end="")

        if self.tying.last_tie_index == -1:
            self.tying.start_guide_points(self.location)
            # Flag that we need to update the guide curve in the LString
            self.tying.tie_needs_updating = True
        else:
            # check location of end point wrt last tie point
            # The last guide point will have been set to the last tie point
            end_pt = np.array(self.location.end)
            last_tie_pt = self.tying.wire_attach.attractor_pts[self.tying.last_tie_index]
            dir_along = end_pt - last_tie_pt
            past = np.dot(dir_along, self.tying.wire_attach.attractor_dir)
            if past > 0.0:
                self.tying.add_next_guide_points(self.location)
                # Flag that we need to update the guide curve in the LString
                self.tying.tie_needs_updating = True
        # Note: actual guide curve will be updated at EndEach hook, not here
        self.logger.info(f"Done updating guide {self.name} {self.tying.last_tie_index} {self.tying.guide_points[-1]}")

    def tie_lstring(self, lstring, index):
        """Insert a SetGuide(...) after position `index` in `lstring`.

        - Removes any immediate following tokens whose .name is in ('&','/','SetGuide').
        - Builds a CSpline from `self.tying.guide_points` and inserts the curve string and length.
        Returns (lstring, removed_count).
        """
        # Nothing to do if we don't have new guide points
        return lstring, 0
        if not self.tying.tie_needs_updating:
            return lstring, 0

        # Build spline and get curve representation (may raise)
        try:
            spline = CSpline(self.tying.guide_points)
            curve_repr = spline.curve(stride_factor=100)
        except Exception as exc:
            raise ValueError("Invalid spline from guide_points") from exc

        # Defensive check for 'nan' in the curve representation (preserve original check intent)
        if "nan" in str(curve_repr):
            raise ValueError("Curve is NaN", self.tying.guide_points)

        # Remove any immediate tokens after index that match the removal set
        removal_names = {"&", "/", "SetGuide"}
        insert_pos = index + 1
        removed_count = 0

        # Remove while the next token exists and matches
        while insert_pos < len(lstring) and getattr(lstring[insert_pos], "name", None) in removal_names:
            del lstring[insert_pos]
            removed_count += 1

        # Insert the new SetGuide token at the computed insert position
        # Upper limit on following the guide curve is the number of wires currently crossed * spacing
        # Once the turtle hits the set guide it will follow it for the length given; so give it the length
        #   of the guide curve
        tie_length = 1.1 * self.tying.last_tie_index * self.tying.wire_attach.spacing
        lstring.insertAt(insert_pos, f"SetGuide({curve_repr}, {tie_length})")

        # Flag that we've added the new guide curve
        self.tying.tie_needs_updating = False

        return lstring, removed_count


class BasicSpur(BasicWood):
    """ A spur can grow leaves and fruit, but does not produce new buds"""
    __count = 0  # For name creation

    def __init__(self, config: BasicWoodConfig, name: str=None):
        super().__init__(config=config)
        if not name:
            self.name = f"Spur_{self.__class__.__count}"
        self.__class__.__count += 1

    def is_add_bud_site(self) -> bool:
        return False

    def create_spur(self):
        return None

    def create_branch(self):
        return None

    def update_guide(self):
        pass

    def tie_lstring(self, lstring, index):
        return lstring, 0


class BasicBranch(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    __count = 0    # Class variable for instance counting

    def __init__(self, config=None, name: str = None):
        # Call BasicWood constructor
        super().__init__(config=config)

        # Set name with automatic numbering
        if not name:
            self.name = f"{self.__class__.__name__}_{BasicBranch.__count}"
        BasicBranch.__count += 1


class BasicTrunk(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    __count = 0    # Class variable for instance counting

    def __init__(self, config=None, name: str = None):
        # Call BasicWood constructor
        super().__init__(config=config)

        # Set name with automatic numbering
        if not name:
            self.name = f"trunk_{BasicTrunk.__count}"
        BasicTrunk.__count += 1

    def angle_wrt_ground(self):
        return 0.0
