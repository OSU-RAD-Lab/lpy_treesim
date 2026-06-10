from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import TreeBranch, BasicWoodConfig, TyingState
from lpy_treesim.tree_models.base_tree.lpy_helper_functions import *


class Spur(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict=None):
        super().__init__(config, copy_from, prototype_dict, )
        return

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        return self.rng.random() < 0.1 * (1 - num_buds_segment / self.growth.max_buds_segment)

    def create_branch(self):
        return None

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        # radius = plant_segment.growth.thickness * simulation_config.thickness_multiplier
        # # return L-Py module directly
        # # from openalea.lpy import newModule
        # return [('@O', [float(radius)])]
        return None


class TertiaryBranch(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict=None):
        super().__init__(config, copy_from, prototype_dict, )
        return

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        if self.rng.random() < 0.005 * self.growth.growth_length * (1 - num_buds_segment / self.growth.max_buds_segment):
            return True

    def create_branch(self):
        if self.rng.random() > 0.8:
            new_ob = Branch(copy_from=self.prototype_dict["side_branch"])
        else:
            new_ob = Spur(copy_from=self.prototype_dict["spur"])
        return new_ob

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class Branch(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict=None):
        super().__init__(config, copy_from, prototype_dict, )
        return

    def is_bud_break(self, num_buds_segment : int) -> bool:
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        if self.rng.random() < 0.2 * (1 - num_buds_segment / self.growth.max_buds_segment):
            return True
        return False

    def create_branch(self):
        try:
            if self.rng.random() > 0.9:
                new_ob = TertiaryBranch(copy_from=self.prototype_dict["side_branch"])
            else:
                new_ob = Spur(copy_from=self.prototype_dict["spur"])
        except:
            return None
        return new_ob

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class Trunk(TreeBranch):
    """Details of the trunk while growing a tree, length, thickness, where to attach them etc"""

    def __init__(self, config=None, copy_from=None, prototype_dict: dict=None):
        super().__init__(config, copy_from, prototype_dict, )
        return

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        if self.rng.random() > 0.05 * self.length / self.growth.max_length * (
            1 - num_buds_segment / self.growth.max_buds_segment
        ):
            return False
        return True

    def create_branch(self):
        if self.rng.random() > 0.1:
            return Branch(copy_from=self.prototype_dict["branch"])

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None

def build_basicwood_prototypes(rng: np.random.Generator):
    # growth_length = 0.1
    basicwood_prototypes = {}

    # Create configs for cleaner prototype setup
    spur_config = BasicWoodConfig(
        max_buds_segment=2,
        max_length=0.1,
        thickness=0.003,
        growth_length=0.05,
        cylinder_length=0.01,
        thickness_increment=0.0005,
        color=(0, 255, 0),
        bud_spacing_age=1,  # Spurs bud every 1 age unit
        curve_x_range=(-0.2, 0.2),  # Tighter bounds for spur curves
        curve_y_range=(-0.2, 0.2),  # Tighter bounds for spur curves
        curve_z_range=(-1, 1),  # Same Z range
        rng=rng
    )

    side_branch_config = BasicWoodConfig(
        max_buds_segment=2,
        max_length=0.25,
        thickness=0.005,
        growth_length=0.05,
        cylinder_length=0.01,
        thickness_increment=0.0005,
        color=(0, 255, 0),
        bud_spacing_age=2,  # Tertiary branches bud every 3 age units
        curve_x_range=(-0.5, 0.5),  # Moderate bounds for tertiary branches
        curve_y_range=(-0.5, 0.5),  # Moderate bounds for tertiary branches
        curve_z_range=(-1, 1),  # Same Z range
        rng=rng
    )

    # Tip the trunk over
    trunk_config = BasicWoodConfig(
        max_buds_segment=5,
        tie_type=TyingState.TyingType.TIE_ALONG,
        max_length=3,
        thickness=0.02,
        thickness_increment=0.00001,
        growth_length=0.1,
        cylinder_length=0.1,
        color=(255, 0, 0),
        bud_spacing_age=2,  # Trunk buds every 4 age units
        curve_x_range=(-0.3, 0.3),  # Conservative bounds for trunk
        curve_y_range=(-0.3, 0.3),  # Conservative bounds for trunk
        curve_z_range=(-0.5, 0.5),  # Tighter Z range for trunk
        prunable=False,
        rng=rng
    )

    # Vertical leaders
    branch_config = BasicWoodConfig(
        max_buds_segment=2,
        tie_type=TyingState.TyingType.TIE_ACROSS,
        max_length=2.5,
        thickness=0.01,
        thickness_increment=0.001,
        growth_length=0.1,
        cylinder_length=0.02,
        color=(255, 150, 0),
        bud_spacing_age=2,  # Branches bud every 2 age units
        curve_x_range=(-0.4, 0.4),  # Moderate bounds for primary branches
        curve_y_range=(-0.4, 0.4),  # Moderate bounds for primary branches
        curve_z_range=(-1, 1),  # Same Z range
        rng=rng
    )

    # Setup prototypes using configs
    basicwood_prototypes["spur"] = Spur(config=spur_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["side_branch"] = TertiaryBranch(config=side_branch_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["trunk"] = Trunk(config=trunk_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["branch"] = Branch(config=branch_config, prototype_dict=basicwood_prototypes)

    return basicwood_prototypes
