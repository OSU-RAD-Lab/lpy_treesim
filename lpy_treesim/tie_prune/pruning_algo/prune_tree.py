from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.primary_heuristic import primary_heuristic_prune
from lpy_treesim.tie_prune.pruning_algo.secondary_heuristic import secondary_heuristic_prune
from lpy_treesim.tie_prune.pruning_algo.prune_spur import spur_prune

def prune_tree(tree_sim_base, 
               branch_hierarchy: dict, 
               map_names_to_branches: dict,
               mark: bool):
    
    # Take out old primary branches
    prune_primary(tree_sim_base, 
                  branch_hierarchy=branch_hierarchy, 
                  map_names_to_branches=map_names_to_branches,
                  mark=mark)

    # Cut short any overly long branches
    # Prunes based on the prune length that is set in 
    prune_length(tree_sim_base,
                 branch_hierarchy=branch_hierarchy, 
                 map_names_to_branches=map_names_to_branches,
                 mark=mark)
    
    
    # Prunes everthing in a set radius based of a referance tree object 
    # Currently pruning primary branches which it should not do
    spur_prune(tree_sim_base=tree_sim_base, 
               branch_hierarchy=branch_hierarchy,
               map_names_to_branches=map_names_to_branches,
               mark=mark)
    
 
    
    # Placeholder fucntion for when it works
    # primary_heuristic_prune(tree_sim_base=tree_sim_base, 
    #                         branch_hierarchy=branch_hierarchy, 
    #                         map_names_to_branches=map_names_to_branches,
    #                         mark=mark)


    
    # Function not working yet
    # secondary_heuristic_prune(tree_sim_base=tree_sim_base, 
    #                           branch_hierarchy=branch_hierarchy, 
    #                           map_names_to_branches=map_names_to_branches,
    #                           mark=mark)