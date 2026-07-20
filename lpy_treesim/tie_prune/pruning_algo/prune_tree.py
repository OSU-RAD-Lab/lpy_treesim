from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_dist import prune_dist

def prune_tree(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    prune_primary(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    
    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    # Prunes everthing in a set radius based of a referance tree object 
    #prune_dist(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)
    

    
    
    
