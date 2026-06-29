"""
Defines the abstract class BasicWood
All wood (trunk, branches, spurs) inherit from this class
"""

from abc import ABC, abstractmethod
import copy
from openalea.plantgl.scenegraph import BezierCurve
from openalea.plantgl.all import Point4Array

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

    def __init__(self, config: BasicWoodConfig):
        # Unique name. Each inherited class tracks a unique number and a counter
        self.name: str = ""

        # Parameters controlling growth are all stored in config
        self.config = config
        if not isinstance(config, BasicWoodConfig):
            raise ValueError("Config should be tree-specific config inherited from BasicWoodConfig")

        # Track end points and orientations
        self.location = LocationState()

        # Tying status - contains what type of tying and attractor points, plus guide points
        self.tying = TyingState(tie_type=config.tie_type,
                                tie_start_dist=config.tie_start_dist,
                                tie_spacing=config.tie_spacing)

        # Growth Variables  - generate variables stochastically
        #   Vigor controls overall growth length/diameter/bud spacing
        vigor = self.config.lpy_rng.normal(0.5, 0.2)
        if vigor < 0.0:
            vigor = 0.0
        if vigor > 1.0:
            vigor = 1.0

        # Use the vigor setting plus some noise to establish growth rate for this branch
        mgl = []
        start_year = 0
        len_max = 0.0
        # The config yearly growth rates don't have to be every year; this makes them every year
        #   and samples from the ranges based on vigor
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
                len_max += mean_value
                start_year += 1

        # This initial growth curve has a bit of noise but points in the z direction (relative coord system)
        self.tying.create_initial_guide_curve(expected_length=len_max,
                                              curve_x_range=config.curve_x_range,
                                              curve_y_range=config.curve_y_range,
                                              lpy_rng=config.lpy_rng)

        # Parameters controlling growth
        self.growth = GrowthState(vigour_level=vigor,
                                  mean_length_growth_per_year=mgl,
                                  num_iter_per_year=self.config.num_iter_per_year,
                                  taper=self.config.taper_amount,
                                  lpy_rng=self.config.lpy_rng)

        # Growth curve before tying
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

    def dist_last_bud(self) -> float:
        """ Where was the last bud added?"""
        if len(self.bud_sites) == 0:
            return 0.0
        return self.bud_sites[-1].dist_along

    def growth_since_last_bud(self) -> float:
        """How far has the branch grown since the last bud site was added?"""
        return self.growth.length - self.dist_last_bud()

    def is_add_bud_site(self) -> bool:
        """This method defines if a bud site should be added here. By default, generates a random number
        for the next bud site location relative to the last and if it's far enough away, generate one
        This is called in base_lpy.lpy every iteration on every branch. """
        # Vigorous branches have buds spaced further apart
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
        """Add a dormant bud at the current site. Once is_add_bud_site returns true, then the next string iteration
        bud_site module will actually generate the bud by calling this method"""
        # Bud labels are the branch name plus the index of the bud along the branch
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

    def grow(self):
        """Called every iteration once the branch is created"""
        self.growth.age_in_iterations += 1
        length_incr = self.growth.get_length_increment()
        # Increment both of these - this tracks both the current and the pruned length
        self.growth.length += length_incr
        self.growth.length_without_pruning += length_incr

    def prune(self, at_length: float) -> list[str]:
        """ Prune some or all of the branch
        Recursively delete bud sites
        Return names of all the deleted objects (for removing from other data structures)"""
        names = []
        bud_keep = []
        for bud in self.bud_sites:
            if bud.dist_along < at_length:
                bud_keep.append(bud)
                continue
            # Takes out the bud's children
            names.extend(bud.prune())
            # Need to remove this bud site as well
            names.append(bud.name)
            # Actually delete the bud
            del bud
        # These are the only ones we're keeping (if any)
        self.bud_sites = bud_keep
        return names

    def add_year(self):
        self.growth.age_in_years += 1

    @abstractmethod
    def create_branch(self):
        """ Returns how a new branch when bud break happens. Eg, if trunk, makes primary branch
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicBranch"""
        pass

    @abstractmethod
    def create_spur(self):
        """ Creates a new spur/fruiting site.
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicSpur"""
        pass

    def initial_growth_curve(self) -> BezierCurve:
        """ If not over-ridden later by tying, generate a growth curve.
        Note: This is relative to the starting direction of the wood object - so z is always 'out' """
        control_point_array = Point4Array(self.tying.guide_points)
        return BezierCurve(control_point_array)

    def contour_curve(self) -> BezierCurve:
        # If you want a non-circular cross section... use lpy_geometry functions to make a contour
        pass

    def update_guide(self):
        """ This updates the guide curve used for tying. If not tying, does nothing
           Tracks the start point and start vector of the branch and only updates if that changes
           This will set the growth curve so that it passes through the desired tying points
           The actual math for this is in tying.py
        """
        if not self.tying.is_tied:
            return

        # Compare self.tying.start_pt_tied and self.start_dir_tied
        if not self.tying.has_moved(start_pt=self.location.start, start_dir=self.location.start_dir):
            return

        print(f"Updating guide for {self.name}")
        # Bend the current guide points to the wire
        self.tying.bend_to_wire(pt_origin=self.location.start, heading=self.location.start_dir, left=self.location.start_left_dir)
        control_point_array = Point4Array(self.tying.guide_points)
        # Will be used the next time SetGuide is called
        self.growth_curve = BezierCurve(control_point_array)

        # If these don't change, then don't need to re-tie
        self.tying.start_pt_tied = self.location.start
        self.tying.start_dir_tied = self.location.start_dir


class BasicSpur(BasicWood):
    """ A spur can grow leaves and fruit, but does not produce new buds"""
    __count = 0  # For name creation

    def __init__(self, config: BasicWoodConfig, name: str = None):
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


class BasicBranch(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    __count = 0    # Class variable for instance counting

    def __init__(self, config: BasicWoodConfig, name: str = None):
        # Call BasicWood constructor
        super().__init__(config=config)

        # Set name with automatic numbering
        if not name:
            self.name = f"{self.__class__.__name__}_{BasicBranch.__count}"
        BasicBranch.__count += 1

    @abstractmethod
    def create_branch(self):
        """ Returns how a new branch when bud break happens. Eg, if trunk, makes primary branch
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicBranch"""
        pass

    @abstractmethod
    def create_spur(self):
        """ Creates a new spur/fruiting site.
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicSpur"""
        pass


class BasicTrunk(BasicWood):
    """Base class for all tree branch types with common initialization logic"""

    __count = 0    # Class variable for instance counting

    def __init__(self, config: BasicWoodConfig, name: str = None):
        # Call BasicWood constructor
        super().__init__(config=config)

        # Set name with automatic numbering
        if not name:
            self.name = f"trunk_{BasicTrunk.__count}"
        BasicTrunk.__count += 1

    @abstractmethod
    def create_branch(self):
        """ Returns how a new branch when bud break happens. Eg, if trunk, makes primary branch
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicBranch"""
        pass

    @abstractmethod
    def create_spur(self):
        """ Creates a new spur/fruiting site.
        These are abstract methods because the type of branch depends on the tree-type
        Do not call directly - BudSite will call on its parent when the bud breaks
        Should return an instance of a class that inherits from BasicSpur"""
        pass

    def angle_wrt_ground(self):
        """ If the trunk is not straight up in the x,z plane, rotate around y by this amount"""
        return 0.0
