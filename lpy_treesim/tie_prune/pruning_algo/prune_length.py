from lpy_treesim.tie_prune.pruning_algo.mark_cuts import mark_cuts

def prune_length(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict, mark: bool):
    """
    Prune branches that exceed their maximum length. Adds some noise to the ending length

    Args:
        branch_hierarchy: Dictionary mapping branch names to lists of child branches
        map_names_to_branches: Dictionary mapping branch names to branch instances

    Returns:
        bool: True if a branch was pruned, False if no eligible branches found

    Note:
        This function processes one branch at a time and returns immediately after
        pruning a single branch. It should be called repeatedly (e.g., in a while loop)
        until no more pruning operations are possible. The cut_from() function handles
        the actual removal of the branch and any dependent substructures from the string.
    """

    # Collect names of all trunks/branches/spurs to be pruned
    # This may indirectly prune bud sites/buds, but not directly
    names_to_x = []
    branches_shortened = [] # Used only for debugging
    branches_pruned = {}
    for name, branch in map_names_to_branches.items():
        
        if "bud" not in name:
            noisy_len = branch.config.noisy_prune_length()
            if branch.growth.length > noisy_len:
                if mark: 
                    branches_pruned[branch.name] = [branch, noisy_len]
                else:
                    names_to_x.extend(branch.prune(noisy_len))
                
                    
                branches_shortened.append(branch.name)
                keep_list = []
                for child in branch_hierarchy[name]:
                    if child.name not in names_to_x:
                        keep_list.append(child)
                branch_hierarchy[name] = keep_list

    if mark:
        if len(branches_pruned)>0:
            location = [branch[0].location.end for branch in branches_pruned.values()]
            LENGTH = len(location)
            type = ["config_length"]*LENGTH
            radius = [0]*LENGTH
            normal_vector = []
            for vector in [branch[0].location.end_dir for branch in branches_pruned.values()]:
                normal_vector.append([[0,0,0],vector])
            name = branches_pruned.keys()
            year = [int(tree_sim_base.current_iteration/tree_sim_base.config.num_iter_per_year)]*LENGTH
            mark_cuts(location, type, radius, normal_vector, year, name) # imported function


    #print(f"Shortened: {branches_shortened}")
    #print(f"{names_to_x}")
    # Now remove any x'd buds etc from the hierarchy
    else:
        for name in names_to_x:
            del branch_hierarchy[name]
            del map_names_to_branches[name]
