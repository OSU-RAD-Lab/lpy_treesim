from lpy_treesim.stochastic_tree import Support, BasicWood, TreeBranch, BasicWoodConfig
import numpy as np
import random as rd
from dataclasses import dataclass
import copy
from openalea.lpy import newmodule
from lpy_treesim.helper import *


class Spur(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict = {}):
        super().__init__(config, copy_from, prototype_dict)

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        return self.rng.random() < 0.1 * (1 - num_buds_segment / self.growth.max_buds_segment)

    def create_branch(self):
        return None

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class Branch(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict = {}):
        super().__init__(config, copy_from, prototype_dict)

    def is_bud_break(self, num_break_buds):
        if num_break_buds >= self.growth.max_buds_segment:
            return False
        return self.rng.random() < 0.5 * (1 - num_break_buds / self.growth.max_buds_segment)

    def create_branch(self):
        if self.rng.random() > 0.8:
            new_ob = TertiaryBranch(copy_from=self.prototype_dict["tertiarybranch"])
        else:
            new_ob = Spur(copy_from=self.prototype_dict["spur"])
        return new_ob

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class Trunk(TreeBranch):
    """Details of the trunk while growing a tree, length, thickness, where to attach them etc"""

    def __init__(self, config=None, copy_from=None, prototype_dict: dict = {}):
        super().__init__(config, copy_from, prototype_dict)

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        if self.rng.random() > 0.1 * (1 - num_buds_segment / self.growth.max_buds_segment):
            return False
        return True

    def create_branch(self):
        if self.rng.random() > 0.8:
            return Spur(copy_from=self.prototype_dict["spur"])
        else:
            return Branch(copy_from=self.prototype_dict["branch"])

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None


class TertiaryBranch(TreeBranch):
    def __init__(self, config=None, copy_from=None, prototype_dict: dict = {}):
        super().__init__(config, copy_from, prototype_dict)

    def is_bud_break(self, num_buds_segment):
        if num_buds_segment >= self.growth.max_buds_segment:
            return False
        return self.rng.random() < 0.5 * (1 - num_buds_segment / self.growth.max_buds_segment)

    def create_branch(self):
        if self.rng.random() > 0.3:
            return None
        else:
            new_ob = Spur(copy_from=self.prototype_dict["spur"])
        return new_ob

    def pre_bud_rule(self, plant_segment, simulation_config):
        return None

    def post_bud_rule(self, plant_segment, simulation_config):
        return None



def build_basicwood_prototypes(rng: np.random.Generator):
    # growth_length = 0.1
    basicwood_prototypes = {}
    # Create configs for cleaner prototype setup
    spur_config = BasicWoodConfig(
        max_buds_segment=5,
        tie_axis=None,
        max_length=0.2,
        thickness=0.003,
        growth_length=0.05,
        cylinder_length=0.05,
        thickness_increment=0.001,
        color=(0, 255, 0),
        bud_spacing_age=2,
        curve_x_range=(-0.2, 0.2),
        curve_y_range=(-0.2, 0.2),
        curve_z_range=(-1, 1),
        prunable=True,
        rng=rng
    )

    branch_config = BasicWoodConfig(
        max_buds_segment=2,
        tie_axis=(1, 0, 0),
        max_length=2.0,
        thickness=0.01,
        growth_length=0.1,
        cylinder_length=0.05,
        thickness_increment=0.001,
        color=(255, 150, 0),
        bud_spacing_age=2,
        curve_x_range=(-0.5, 0.5),
        curve_y_range=(-0.5, 0.5),
        curve_z_range=(-1, 1),
        prunable=True,
        rng=rng
    )

    trunk_config = BasicWoodConfig(
        max_buds_segment=5,
        tie_axis=None,
        max_length=4,
        thickness=0.1,
        growth_length=0.1,
        cylinder_length=0.05,
        thickness_increment=0.01,
        color=(255, 0, 0),
        bud_spacing_age=2,
        curve_x_range=(-1, 1),
        curve_y_range=(-0.15, 0.15),
        curve_z_range=(0, 10),
        prunable=False,
        rng=rng
    )

    tertiarybranch_config = BasicWoodConfig(
        max_buds_segment=5,
        tie_axis=None,
        max_length=0.3,
        thickness=0.003,
        growth_length=0.05,
        cylinder_length=0.05,
        thickness_increment=0.001,
        color=(0, 255, 0),
        bud_spacing_age=2,
        curve_x_range=(-0.5, 0.5),
        curve_y_range=(-0.5, 0.5),
        curve_z_range=(-1, 1),
        prunable=True,
        rng=rng
    )

    # Setup prototypes using configs
    basicwood_prototypes["spur"] = Spur(config=spur_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["branch"] = Branch(config=branch_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["trunk"] = Trunk(config=trunk_config, prototype_dict=basicwood_prototypes)
    basicwood_prototypes["tertiarybranch"] = TertiaryBranch(config=tertiarybranch_config, prototype_dict=basicwood_prototypes)
    return basicwood_prototypes
