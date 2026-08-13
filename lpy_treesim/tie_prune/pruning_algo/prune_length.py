#from lpy_treesim.tie_prune.tie_prune_simulation_base import TreeSimulationBase


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
    branches_shortened = []
    branches_to_prune = []
    for name, branch in map_names_to_branches.items():
        if "bud" in name:
            continue
        noisy_len = branch.config.noisy_prune_length()
        if branch.growth.length > noisy_len:
            
            if not mark: 
                names_to_x.extend(branch.prune(noisy_len))
                
            branches_shortened.append(branch.name)
            keep_list = []
            for child in branch_hierarchy[name]:
                if child.name not in names_to_x:
                    keep_list.append(child)
            branch_hierarchy[name] = keep_list

    #print(f"Shortened: {branches_shortened}")
    #print(f"{names_to_x}")
    # Now remove any x'd buds etc from the hierarchy
    if not mark:
        for name in names_to_x:
            del branch_hierarchy[name]
            del map_names_to_branches[name]
