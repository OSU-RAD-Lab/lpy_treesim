"""
Defines a bud site; bud sites mark where on the branch/trunk spurs/branches can grow
"""

from abc import ABC
import numpy as np
from dataclasses import dataclass
from enum import Enum
from openalea.plantgl.all import Vector3

import logging


@dataclass
class BudSite(ABC):
    """ Potential bud site on branch. These should be positioned (roughly) evenly along the branch at the
    desired spacing. The bud can be vegetative or fruiting or mixed - if mixed, will produce a fruiting bud
    followed by a vegetative bud with some probability (as opposed to just a vegetative or fruiting bud)
    If a bud is marked as dormant because it has nothing growihg out of it (yet)
    Dormant buds can transition to vegetative or fruiting or mixed buds with some probability, spawning either a branch or a spur or both
    Pruned buds mark where wood was pruned; they can be resurrected by turning them back to dormant"""

    class BudType(Enum):
        VEGETATIVE = "vegetative"
        FRUITING = "fruiting"
        MIXED = "mixed"
        DORMANT = "dormant"
        PRUNED = "pruned"

    bud_state: BudType = BudType.DORMANT
    bud_break_probabilities: tuple = (0.1, 0.3, 0.25)  # eg, will turn vegetative with 0.1 prob, fruiting w 0.3 - 0.1
    bud_angle_probabilities: dict = None              # Bud angle relative to branch; angle may depend on type of bud
    bud_angle_around: float = 0.0
    bud_angle_from_parent: float = 90.0

    # These are filled in when the string is interpreted
    start_loc: Vector3 = Vector3(0, 0, 0)
    start_dir: Vector3 = Vector3(0, 0, 0)

    # Unique name for bud site - parent name + bud and id
    name: str = ""

    # This tracks with the current branch's length when the bud was spawned. It is used to determine
    #  when the next bud will spawn
    dist_along: float = 0.0

    age_year: int = 0

    # Parent branch - used to call create branch/spur on parent
    #   Note - can't declare type because that creates a circular reference
    wood_parent: object = None
    branch_child: object = None
    spur_child: object = None

    # Random number to use - this is here for repeatability
    lpy_rng: np.random.Generator = None

    def is_dormant_or_pruned(self):
        return self.bud_state is BudSite.BudType.DORMANT or self.bud_state is BudSite.BudType.PRUNED

    def is_bud_break(self):
        """ If bud break... will create the branch in the create_branch method"""
        if self.bud_state is not BudSite.BudType.DORMANT:
            return False
        # Controls when the buds break
        prob = self.lpy_rng.uniform(0.0, 1.0)
        if prob < self.bud_break_probabilities[0]:
            self.bud_state = BudSite.BudType.VEGETATIVE
        elif prob < self.bud_break_probabilities[1]:
            self.bud_state = BudSite.BudType.FRUITING
        elif prob < self.bud_break_probabilities[2]:
            self.bud_state = BudSite.BudType.MIXED
        else:
            return False
        # Breaking out of dormancy
        bud_angle_range = self.bud_angle_probabilities[self.bud_state]
        self.bud_angle_from_parent = self.lpy_rng.uniform(bud_angle_range[0], bud_angle_range[1])
        return True

    def create_branch(self):
        """ If vegetative or mixed, will create a branch.
        These (eventually) call the create_branch/spur methods on the iherited class """
        if self.bud_state is BudSite.BudType.VEGETATIVE or self.bud_state is BudSite.BudType.MIXED:
            self.branch_child = self.wood_parent.create_branch()
            return self.branch_child
        return None

    def create_spur(self):
        """If fruiting or mixed, will create a spur"""
        if self.bud_state is BudSite.BudType.FRUITING or self.bud_state is BudSite.BudType.MIXED:
            self.spur_child = self.wood_parent.create_spur()
            return self.spur_child
        return None

    def add_year(self):
        self.age_year += 1

    def start_diameter(self):
        t = self.dist_along / self.wood_parent.growth.length
        if t < 0.0 or t > 1.0:
            print(f"bad t {t}")
        parent_diameter = self.wood_parent.growth.get_diameter(t)
        if parent_diameter <= 0.0:
            print(f"Bad bud {parent_diameter}")
            parent_diameter = 0.001
        return 0.1 * parent_diameter

    def end_diameter(self):
        return 0.0001

    def draw_length(self):
        return 2 * self.wood_parent.growth.get_diameter(self.dist_along)
