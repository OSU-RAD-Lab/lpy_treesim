#from lpy_treesim.tie_prune.tie_prune_simulation_base import TreeSimulationBase
from lpy_treesim.tree_models.base_tree.tree_wood_prototypes import BasicWood

def prune_primary(tree_sim_base, 
                  branch_hierarchy: dict, 
                  map_names_to_branches: dict):
    """
    Prune old branches that exceed the age_in_iterations threshold and haven't been tied to wires.

    This function implements the pruning strategy for the tree training simulation.
    It identifies branches that have grown too old (exceeding the pruning age_in_iterations threshold)
    but haven't been successfully tied to trellis wires.

    Note: Removing items from the dictionary will cause WoodStart, SpurStart, and BudStart to x out the
        string the next iteration. See base_lpy.lpy

    The pruning criteria are:
    1. Branch age_in_iterations exceeds the configured pruning threshold
    2. Branch has not been tied to any trellis wire
    3. Branch has not already been marked for cutting
    4. Branch is prunable (respects the prunable flag)

    When a branch meets all criteria, it is:
    - Removed from branch_hierarchy
    - Removed from parent's children list
    - Removed from the map names dictionary

    Note that the bud site that generated the pruned branch will remain and be marked pruned

    Args:
        branch_hierarchy: Dictionary mapping branch names to lists of child branches
        map_names_to_branches: Dictionary mapping branch names to pointers to branches
    """

    # Collect names of buds/branches/spurs to be pruned
    buds_to_prune = []
    for branch_name, branch_children in branch_hierarchy.items():
        if "trunk" not in branch_name:
            continue

        # Buds on trunk
        for bud in branch_children:
            if not "bud" in bud.name:
                # This shouldn't happen, but...
                continue
            if not bud.branch_child:
                # Skip buds that don't have branches
                continue

            branch: BasicWood = bud.branch_child
            age_exceeds_threshold = branch.growth.age_in_iterations > tree_sim_base.config.pruning_age_threshold
            not_tied_to_wire = not branch.tying.is_tied
            prune_by_age = branch.config.remove_at_age

            # Prune if all criteria are met
            if age_exceeds_threshold and not_tied_to_wire and prune_by_age:
                buds_to_prune.append(bud)

    # Now add the bud names (and all the bud's branch children) to the list
    names_to_x = []
    for bud in buds_to_prune:
        # This removes spurs/branches, marks the bud as pruned, and recursively removes the children
        names_to_x.extend(bud.prune())

    # Now remove the names from branch hierarchy
    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]

    '''
    # In the next iteration WoodStart etc will be replaced with % and cut out
    print(f"Left: ")
    for key in map_names_to_branches.keys():
        if "primary" in key and "bud" not in key:
            pass
            print(f"{key}")
    '''