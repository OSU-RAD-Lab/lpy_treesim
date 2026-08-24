"""
V-trellis tree architectures (apples)
  Support trellis tips out at 15 degrees.
  Main trunk is tied across all the wires
  Primary support branches are tied down horizontally to the wires
  Tertiary grow in all directions
  Uses default spur
  TODOS:
"""
import numpy as np
from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicSpur, BasicBranch, BasicTrunk
from lpy_treesim.tree_models.base_tree.basic_wood_config import BasicWoodConfig
from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig


class TertiaryBranch(BasicBranch):
    def __init__(self, config):
        super().__init__(config)

    def create_branch(self):
        # Create another side branch
        new_branch = TertiaryBranch(config=self.config.configs_dict["tertiary_branch"])
        return new_branch

    def create_spur(self):
        new_spur = BasicSpur(config=self.config.configs_dict["spur"])
        return new_spur


class PrimaryBranch(BasicBranch):
    def __init__(self, config):
        super().__init__(config)

    def create_branch(self):
        # Create another side branch
        new_branch = TertiaryBranch(config=self.config.configs_dict["tertiary_branch"])
        return new_branch

    def create_spur(self):
        new_spur = BasicSpur(config=self.config.configs_dict["spur"])
        return new_spur


class Trunk(BasicTrunk):
    """Details of the trunk while growing a tree, length, thickness, where to attach them etc"""

    def __init__(self, config):
        super().__init__(config)

    def create_branch(self):
        new_branch = PrimaryBranch(config=self.config.configs_dict["primary_branch"])
        return new_branch

    def create_spur(self):
        new_spur = BasicSpur(config=self.config.configs_dict["spur"])
        return new_spur

    def angle_wrt_ground(self):
        return 45


def build_basicwood_prototypes(lpy_rng: np.random.Generator = None, sim_config: SimulationConfig = None):
    """ For apples, the main trunk goes up and is trimmed at the top wire
    The primary branches grow fairly rapidly while the tertiary are slower
    Phyllotaxis: The ussual 2/5 spiral (137.5 degrees)
    Bud spacing is 1.5 to 2 inches. (Red delicious & macoun have 0.5-1)
    Growth rates: Vigorous (water sprouts) can grow several feet in one year,
    with most growth slower"""

    if lpy_rng is None:
        lpy_rng = np.random.default_rng()

    bud_angle_probs = {BudSite.BudType.VEGETATIVE: (30, 45),
                       BudSite.BudType.FRUITING: (30, 45),
                       BudSite.BudType.MIXED: (30, 45)}

    # Create configs for cleaner prototype setup
    spur_config = BasicWoodConfig(bud_spacing_range=(0.001, 0.002),  # Use for spacing each age_in_years's fruit location
                                  yearly_growth_range=[(1, 0.1, 0.15), (3, 0.0025, 0.05)],  # Grows 1-2 inches per age_in_years
                                  taper_amount=0.3,  # Stubby
                                  curve_x_range=(-0.02, 0.02),
                                  curve_y_range=(-0.02, 0.02),
                                  num_iter_per_year=sim_config.num_iter_per_year,
                                  lpy_rng=lpy_rng,
                                  #prune_length = 0.1 # Control on spur length (temporary?)
                                  )

    tertiary_branch_config = BasicWoodConfig(bud_spacing_range=(0.01, 0.02),  # Slightly less than the vertical leaders
                                             yearly_growth_range=[(1, 0.02, 0.04), (3, 0.0025, 0.05)],  # 4-12 inches, dropping to 1-2 inches
                                             taper_amount=0.3,    # Changed from 0.1, too skinny
                                             bud_angle_probs=bud_angle_probs,
                                             bud_break_probs=(0.1, 0.6, 0.7),  # Veg, fruit, mixed
                                             curve_x_range=(-0.1, 0.1),
                                             curve_y_range=(-0.1, 0.1),
                                             num_iter_per_year=sim_config.num_iter_per_year,
                                             lpy_rng=lpy_rng)

    tie_spacing = 0.5 * sim_config.width() / (sim_config.n_along + 0.5) # 1/2 of tree divided by number of tie points plus 0.5
    primary_branch_config = BasicWoodConfig(bud_spacing_range=(0.0254, 0.0508),  # 1-2 inches
                                            # 12-36 inches per age_in_years, tapering off
                                            yearly_growth_range=[(1, 0.3, 0.9), (2, 0.25, 0.7), (3, 0.1, 0.3), (4, 0.05, 0.15)],
                                            remove_at_age=True,
                                            taper_amount=0.2,   # Not too skinny
                                            prune_length=sim_config.width(),
                                            bud_angle_probs=bud_angle_probs,
                                            bud_break_probs=(0.6, 0.7, 0.8),   # Most buds break as vegetative
                                            tie_type=TyingState.TyingType.TIE_ALONG,
                                            tie_start_dist=0.5 * tie_spacing,  # 18 inches from first wire
                                            tie_spacing=tie_spacing,
                                            curve_x_range=(-0.1, 0.1), # changed from 0.2 to get less wiggly branches
                                            curve_y_range=(-0.1, 0.1),
                                            num_iter_per_year=sim_config.num_iter_per_year,
                                            lpy_rng=lpy_rng)

    trunk_config = BasicWoodConfig(bud_spacing_range=(0.0254, 0.0508),  # 1-2 inches
                                   # 24-36 inches per age_in_years, tapering off
                                   yearly_growth_range=[(1, 0.6, 0.9), (4, 0.5, 0.7), (5, 0.15, 0.3), (6, 0.05, 0.15)],
                                   taper_amount=0.4,  # Not too skinny
                                   prune_length=0.9 * (sim_config.end_height()), # Changed from 1.1, trunk was sticking up too much
                                   bud_angle_probs=bud_angle_probs,
                                   bud_break_probs=(0.6, 0.7, 0.8),  # Most buds break as vegetative
                                   tie_type=TyingState.TyingType.TIE_ACROSS,
                                   curve_x_range=(-0.2, 0.2),
                                   curve_y_range=(-0.2, 0.2),
                                   tie_start_dist=sim_config.start_height,  # 20 inches from ground to first wire
                                   tie_spacing=sim_config.spacing_wires,      # 12 inches between wires
                                   num_iter_per_year=sim_config.num_iter_per_year,
                                   lpy_rng=lpy_rng)

    # Setup prototypes using configs
    basicwood_prototypes = {"spur": spur_config,
                            "tertiary_branch": tertiary_branch_config,
                            "primary_branch": primary_branch_config,
                            "trunk": trunk_config}

    for item in basicwood_prototypes.values():
        item.configs_dict = basicwood_prototypes
        # Pad out the yearly growth rate so that it matches the total number of simulation years
        if item.yearly_growth_range[-1][0] < sim_config.n_years:
            item.yearly_growth_range.append((sim_config.n_years, item.yearly_growth_range[-1][1], item.yearly_growth_range[-1][2]))

    return basicwood_prototypes
