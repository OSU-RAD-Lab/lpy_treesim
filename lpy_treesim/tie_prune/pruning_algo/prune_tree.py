from lpy_treesim.tie_prune.pruning_algo.prune_random import prune_random
from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_random import prune_random
from lpy_treesim.tie_prune.pruning_algo.prune_dist import prune_dist
from lpy_treesim.tie_prune.pruning_algo.three_d import run_three_d
from lpy_treesim.tie_prune.pruning_algo.prune_at_end import end_prune

def prune_tree(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):

    
    # Hide the list from previous years so prune_length doesn't bug over it
    temp_marked = map_names_to_branches.pop('marked_locations', None)

    prune_primary(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    
    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    prune_dist(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    
    # randomly prune to test pruning (playground for testing/learning purposes - comment out for actual algorithm)
    prune_random(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    # Catch the returned list directly from the function 
    marked_locations = end_prune(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    
    if marked_locations:
        run_three_d(prune_locations=marked_locations, k_neighbors=2, distance_threshold=0.1)
        map_names_to_branches['marked_locations'] = marked_locations
    elif temp_marked:
        # If no new buds were found this year, put the old list back so make_n_trees.py can still find it at the end
        map_names_to_branches['marked_locations'] = temp_marked
    else:
        print("No marked locations found for 3D evaluation.")
    
    
