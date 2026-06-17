from lpy_treesim.tie_prune.tying import TyingState
from lpy_treesim.tree_models.base_tree.bud_site import BudSite
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicSpur, BasicBranch, BasicTrunk
from lpy_treesim.lpy_functions.lpy_helper_functions import *
from lpy_treesim.tree_generation.tree_naming_convention import TreeNamingConvention
from lpy_treesim.tie_prune.tie_prune_simulation_base import SimulationConfig


class SideBranch(BasicBranch):
    def __init__(self, config):
        super().__init__(config)

    def create_branch(self):
        # Create another side branch
        new_branch = SideBranch(config=self.config.configs_dict["side_branch"])
        return new_branch

    def create_spur(self):
        new_spur = BasicSpur(config=self.config.configs_dict["spur"])
        return new_spur

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class PrimaryBranch(BasicBranch):
    def __init__(self, config):
        super().__init__(config)

    def create_branch(self):
        # Create another side branch
        new_branch = SideBranch(config=self.config.configs_dict["side_branch"])
        return new_branch

    def create_spur(self):
        new_spur = BasicSpur(config=self.config.configs_dict["spur"])
        return new_spur

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


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

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None

    def angle_wrt_ground(self):
        return 45


def build_basicwood_prototypes(lpy_rng: np.random.Generator = None, sim_config: SimulationConfig = None):
    """ For cherries, we have the trunk which generates vertical leaders (side branches that turn into
         vertical leaders). Each vertical leader generates side branches fairly regularly; these side
         branches are mostly spurs (just fruiting sites) with the occaisionally grow too much branch.
        Side branches can generate more side branches, although with diminishing growth rates"""

    if lpy_rng is None:
        lpy_rng = np.random.default_rng()

    bud_angle_probs = {BudSite.BudType.VEGETATIVE: (15, 35),
                       BudSite.BudType.FRUITING: (30, 50),
                       BudSite.BudType.MIXED: (15, 50)}

    # Create configs for cleaner prototype setup
    spur_config = BasicWoodConfig(bud_spacing_range=(0.001, 0.002),  # Use for spacing each age_in_years's fruit location
                                  yearly_growth_range=[(1, 0.1, 0.15), (3, 0.0025, 0.05)],  # Grows 1-2 inches per age_in_years
                                  taper_amount=0.01,  # Ends in a point
                                  curve_x_range=(-0.02, 0.02),
                                  curve_y_range=(-0.02, 0.02),
                                  color=TreeNamingConvention.semantic_color("spur"),
                                  num_iter_per_year=sim_config.num_iter_per_year,
                                  lpy_rng=lpy_rng)

    side_branch_config = BasicWoodConfig(bud_spacing_range=(0.01, 0.02),  # Slightly less than the vertical leaders
                                         yearly_growth_range=[(1, 0.025, 0.05), (3, 0.0025, 0.05)],  # 4-12 inches, dropping to 1-2 inches
                                         taper_amount=0.1,    # Gets skinny
                                         prune_length=0.15,   # Prune past 6 inches
                                         bud_angle_probs=bud_angle_probs,
                                         bud_break_probs=(0.05, 0.5, 0.55),
                                         curve_x_range=(-0.1, 0.1),
                                         curve_y_range=(-0.1, 0.1),
                                         color=TreeNamingConvention.semantic_color("branch"),
                                         num_iter_per_year=sim_config.num_iter_per_year,
                                         lpy_rng=lpy_rng)

    primary_branch_config = BasicWoodConfig(bud_spacing_range=(0.0254, 0.0508),  # 1-2 inches
                                            # 24-36 inches per age_in_years, tapering off
                                            yearly_growth_range=[(1, 0.6, 0.9), (2, 0.5, 0.7), (3, 0.15, 0.3), (4, 0.05, 0.15)],
                                            taper_amount=0.2,   # Not too skinny
                                            prune_length=1.2 * sim_config.end_height(),
                                            bud_angle_probs=bud_angle_probs,
                                            bud_break_probs=(0.2, 0.6, 0.8),   # Most buds break as fruiting
                                            tie_type=TyingState.TyingType.TIE_ACROSS,
                                            curve_x_range=(-0.2, 0.2),
                                            curve_y_range=(-0.2, 0.2),
                                            color=TreeNamingConvention.semantic_color("branch"),
                                            num_iter_per_year=sim_config.num_iter_per_year,
                                            lpy_rng=lpy_rng)

    trunk_config = BasicWoodConfig(bud_spacing_range=(0.0254, 0.0508),  # 1-2 inches
                                   # 24-36 inches per age_in_years, tapering off
                                   yearly_growth_range=[(1, 0.6, 0.9), (2, 0.5, 0.7), (3, 0.15, 0.3), (4, 0.05, 0.15)],
                                   taper_amount=0.3,  # Not too skinny
                                   prune_length=1.1 * (sim_config.x_right - sim_config.x_left),
                                   bud_angle_probs=bud_angle_probs,
                                   bud_break_probs=(0.6, 0.7, 0.8),  # Most buds break as vegetative
                                   tie_type=TyingState.TyingType.TIE_ALONG,
                                   curve_x_range=(-0.2, 0.2),
                                   curve_y_range=(-0.2, 0.2),
                                   color=TreeNamingConvention.semantic_color("trunk"),
                                   num_iter_per_year=sim_config.num_iter_per_year,
                                   lpy_rng=lpy_rng)

    # Setup prototypes using configs
    basicwood_prototypes = {"spur": spur_config,
                            "side_branch": side_branch_config,
                            "primary_branch": primary_branch_config,
                            "trunk": trunk_config}

    for item in basicwood_prototypes.values():
        item.configs_dict = basicwood_prototypes
        # Pad out the yearly growth rate so that it matches the total number of simulation years
        if item.yearly_growth_range[-1][0] < sim_config.n_years:
            item.yearly_growth_range.append((sim_config.n_years, item.yearly_growth_range[-1][1], item.yearly_growth_range[-1][2]))

    return basicwood_prototypes
