from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_random import prune_random
from lpy_treesim.tie_prune.pruning_algo.prune_dist import prune_dist

def prune_tree(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    # Added this code. This makes it so prune_length doesn't crash trying to read the .config
    temp_locs = map_names_to_branches.pop('prune_locations_for_3d', None)
    temp_rad = map_names_to_branches.pop('prune_radius_for_3d', None)

    prune_primary(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    
    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    #prune_dist(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    
    # randomly prune to test pruning (playground for testing/learning purposes - comment out for actual algorithm)
    # prune_random(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
