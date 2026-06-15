"""
Defines the abstract class BasicWood along with helper classes LocationState, GrowthState, InfoState and TyingState
"""

from abc import ABC, abstractmethod
from openalea.plantgl.all import *
import copy
import numpy as np
from openalea.plantgl.scenegraph.cspline import CSpline
import collections
from dataclasses import dataclass
from enum import Enum

import logging


@dataclass
class BudSite(ABC):
    """ Potential bud site on branch. These should be positioned (roughly) evenly along the branch at the
    desired spacing. The bud can be vegetative or fruiting or mixed - if mixed, will produce a fruiting bud
    followed by a vegetative bud with some probability (as opposed to just a vegetative or fruiting bud)
    If a bud is marked as dormant than it has nothing growihg out of it (yet)
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
    bud_angle_from_parent: float = 0.0

    # Unique name for bud site - parent name + bud and id
    name: str = ""

    # This tracks with the current branch's length when the bud was spawned. It is used to determine
    #  when the next bud will spawn
    dist_along: float = 0.0

    # Parent branch - used to call create branch/spur on parent
    wood_parent = None
    branch_child = None
    spur_child = None

    # Random number to use - this is here for repeatability
    lpy_rng: np.random.Generator = None

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
        """ If vegetative or mixed, will create a branch"""
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
