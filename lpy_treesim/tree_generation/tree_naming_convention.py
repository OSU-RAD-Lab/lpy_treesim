"""
Part naming management utilities for tree structures

This module provides utilities for assigning unique names to tree structures to support semantic labeling
in isaac sim and else where.

Hierarchical naming structure

root_stock - the base root stock component (if tree is grafted)
trunk - the trunk(s)
branchL01, branchL02 etc - primary, secondary branches (determined by branching structure)
spur
leaf
flower
fruit

For each item above, instances are labeled -n, eg, trunk-Id00 is the first trunk, trunk-Id01 is the second trunk
For branches/spurs, the hierarchy is in the name. Eg, a spur on a second branch would have a full name of
trunk-Id01_branchL0-Id10_branchL1-Id15_spur-Id30

and a "short" name of
spur-Id30

When building a tree the intent is to have bi-directional and "flattened" dictionaries. Eg, give me a list of all of
the spurs, give me all branches of level 1, give me the parent trunk for a branch, etc.

See also skeleton_convention for defining a skeleton for each tree part (centerline plus radius) that also links face
and vertex ids to mesh components
"""

import itertools
import json


class TreeNamingConvention:
    part_names = ["rootstock", "trunk", "branch", "spur", "leaf", "flower", "fruit", "budsite"]
    ROOT_STOCK = 0
    TRUNK = 1
    BRANCH = 2
    SPUR = 3
    LEAF = 4
    FLOWER = 5
    FRUIT = 6
    BUDSITE = 7

    # For semantic labeling by part
    _component_colors={"rootstock":(50, 50, 50),
                       "trunk":(50, 50, 255),
                       "branch":((50, 220, 50), (40, 190, 40), (30, 160, 30), (20, 130, 20)),
                       "spur":(255, 50, 50),
                       "bud":(255, 255, 0)}

    def __init__(self):
        pass

    @staticmethod
    def semantic_color(name):
        if "trunk" in name:
            return TreeNamingConvention._component_colors["trunk"]
        elif "branch" in name:
            if "L" in name:
                split_name = name.split('-')
                indx = min(int(split_name[1]), len(TreeNamingConvention._component_colors["branch"]) - 1)
                return TreeNamingConvention._component_colors["branch"][indx]
            else:
                return TreeNamingConvention._component_colors["branch"][0]
        elif "spur" in name:
            return TreeNamingConvention._component_colors["spur"]
        elif "bud":
            return TreeNamingConvention._component_colors["bud"]
        else:
            print(f"No known semantic name {name}")
        return (255, 255, 255)

    def instance_color(self, name):
        if "trunk" in name:
            col = TreeNamingConvention._component_colors["trunk"]
            split_name = name.split('-')
            indx = int(split_name[-1])
            # Once the tree is computed, this should be the maximum number of branches
            blue = 150 + indx * 100 // (self.current_trunk_id + 1)
            return (col[0], col[1], blue)
        elif "branch" in name:
            if "L" in name:
                split_name = name.split('-')
                level_indx = min(int(split_name[1]), len(TreeNamingConvention._component_colors["branch"]) - 1)
                id_indx = int(split_name[-1])
                col = TreeNamingConvention._component_colors["branch"][level_indx]
                red_blue = 0
                while id_indx > 25:
                    red_blue += 1
                    id_indx -= 25
                green = id_indx
                return (col[0] + red_blue, col[1] + green, col[2] + red_blue)
            else:
                return TreeNamingConvention._component_colors["branch"][0]
        elif "spur" in name:
            col = TreeNamingConvention._component_colors["spur"]
            split_name = name.split('-')
            spur_id = int(split_name[-1])
            green_blue = 0
            while spur_id > 100:
                green_blue += 1
                spur_id -= 100

            return (spur_id * 2, col[1] + green_blue, col[2] + green_blue)
        else:
            print(f"No known semantic name {name}")
        return (255, 255, 255)

    @staticmethod
    def _root_key():
        return TreeNamingConvention.part_names[TreeNamingConvention.ROOT_STOCK]

    @staticmethod
    def _trunk_key():
        return TreeNamingConvention.part_names[TreeNamingConvention.TRUNK]

    @staticmethod
    def _branch_key():
        return TreeNamingConvention.part_names[TreeNamingConvention.BRANCH]

    @staticmethod
    def _spur_key():
        return TreeNamingConvention.part_names[TreeNamingConvention.SPUR]

    def trunk_parts(self):
        return self.part_list[TreeNamingConvention._trunk_key()]

    def branch_parts(self):
        return self.part_list[TreeNamingConvention._branch_key()]

    @staticmethod
    def _rootstock_name(has_root_stock: bool)->str:
        if has_root_stock:
            root_stock_name = f"{TreeNamingConvention._root_key()}"
        else:
            root_stock_name = "NoRootStock"
        return root_stock_name

    @staticmethod
    def rootstock_full_name(has_root_stock: bool)->str:
        root_stock_name = TreeNamingConvention._rootstock_name(has_root_stock)
        return root_stock_name

    @staticmethod
    def _trunk_name(trunk_id:int)->str:
        trunk_name = f"{TreeNamingConvention._trunk_key()}Id-{trunk_id}"
        return trunk_name

    @staticmethod
    def trunk_full_name(has_root_stock: bool, trunk_id:int)->str:
        root_stock_name = TreeNamingConvention.rootstock_full_name(has_root_stock)
        trunk_name = TreeNamingConvention._trunk_name(trunk_id=trunk_id)
        return f"{root_stock_name}_{trunk_name}"

    @staticmethod
    def trunk_usd_name(trunk_id:int)->str:
        trunk_usd_name = f"{TreeNamingConvention._trunk_name(trunk_id=trunk_id)}"
        return trunk_usd_name

    @staticmethod
    def _branch_name(branch_level:int, branch_id:int)->str:
        branch_name = f"{TreeNamingConvention._branch_key()}L-{branch_level}-Id-{branch_id:03d}"
        return branch_name

    @staticmethod
    def branch_full_name(has_root_stock: bool, trunk_id: int, branch_and_parent_ids: list)->str:
        root_stock_name = TreeNamingConvention.rootstock_full_name(has_root_stock)
        trunk_name = f"{TreeNamingConvention._trunk_key()}Id{trunk_id}"
        build_name = f"{root_stock_name}_{trunk_name}"
        for level, id in enumerate(branch_and_parent_ids):
            branch_name = TreeNamingConvention._branch_name(level, id)
            build_name = f"{build_name}_{branch_name}"

        return build_name

    @staticmethod
    def branch_usd_name(trunk_id: int, branch_and_parent_ids: list)->str:
        trunk_usd_name = TreeNamingConvention.trunk_usd_name(trunk_id=trunk_id)
        build_usd_name = f"{trunk_usd_name}"
        for level, id in enumerate(branch_and_parent_ids):
            branch_name = TreeNamingConvention._branch_name(level, id)
            build_usd_name = f"{build_usd_name}/{branch_name}"
        return build_usd_name

    @staticmethod
    def _spur_name(spur_id:int)->str:
        spur_name = f"spurId-{spur_id:04d}"
        return spur_name

    @staticmethod
    def spur_full_name(has_root_stock: bool, trunk_id: int, branch_and_parent_ids: list, spur_id:int)->str:
        root_stock_name = TreeNamingConvention.rootstock_full_name(has_root_stock)
        trunk_name = f"{TreeNamingConvention._trunk_key()}Id{trunk_id}"
        build_name = f"{root_stock_name}_{trunk_name}"
        for level, id in enumerate(branch_and_parent_ids):
            branch_name = TreeNamingConvention._branch_name(level, id)
            build_name = f"{build_name}_{branch_name}"

        spur_name = TreeNamingConvention._spur_name(spur_id=spur_id)
        build_name = f"{build_name}_{spur_name}"
        return build_name

    @staticmethod
    def spur_usd_name(trunk_id: int, branch_and_parent_ids: list, spur_id:int)->str:
        trunk_usd_name = TreeNamingConvention.trunk_usd_name(trunk_id=trunk_id)
        build_usd_name = f"{trunk_usd_name}"
        for level, id in enumerate(branch_and_parent_ids):
            branch_name = TreeNamingConvention._branch_name(level, id)
            build_usd_name = f"{build_usd_name}/{branch_name}"

        spur_name = TreeNamingConvention._spur_name(spur_id=spur_id)
        build_usd_name = f"{build_usd_name}/{spur_name}"
        return build_usd_name

    @staticmethod
    def get_root_stock_id(part_dict: dict)->int:
        return part_dict[TreeNamingConvention._root_key()]

    @staticmethod
    def get_trunk_id(part_dict: dict)->int:
        return part_dict[TreeNamingConvention._trunk_key()]
    
    @staticmethod
    def get_branch_level(full_name: str)->int:
        # TODO: Test this method
        parent_parts = full_name.split("_")
        last_name = parent_parts[-1]
        if "spur" in last_name:
            last_name = parent_parts[-2]

        if not "branch" in last_name:
            return -1  # trunk
        
        name_parts = last_name.split("-")
        level_str = name_parts[1]
        return int(level_str)
