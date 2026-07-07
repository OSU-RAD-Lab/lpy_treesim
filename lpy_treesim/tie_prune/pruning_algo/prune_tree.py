from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_random import prune_random

def prune_tree(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    ### These edit branch_hierarchy and map_names_to_branches in place to remove the branches/spurs etc ###
    
    # Take out old primary branches
    prune_primary(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    # randomly prune to test pruning (playground for testing/learning purposes - comment out for actual algorithm)
    prune_random(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)