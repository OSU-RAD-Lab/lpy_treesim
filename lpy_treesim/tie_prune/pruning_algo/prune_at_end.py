from lpy_treesim.tie_prune.pruning_algo.radius_of_neighbors_class import NeighborsDataStructure
from lpy_treesim.tie_prune.pruning_algo.prune_dist import dist_prune
from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
import pandas as pd
from pathlib import Path

# Pruning that only happens on the final iteration/year (iteration 167)
# NONE OF THE PRUNING RULES IN prune_tree.py WILL BE APPLIED UNLESS FUNCTION IS CALLED HERE

# changed from (self, branch_hierarchy: dict, map_names_to_branches: dict):
def end_prune(tree_sim_base, 
              branch_hierarchy: dict, 
              map_names_to_branches: dict):
    PRINT = True
    RADIUS = 0.1
    df = pd.DataFrame(columns=['marked_x', 'marked_y', 'marked_z', 'radius', 'name'])
    if PRINT: print("[end_prune] Function Called, end_prune() running...")

    prune_primary(tree_sim_base=tree_sim_base, 
                  branch_hierarchy=branch_hierarchy, 
                  map_names_to_branches=map_names_to_branches)
    
    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, 
                 branch_hierarchy=branch_hierarchy, 
                 map_names_to_branches=map_names_to_branches)

    #currently the same code that is in prune_tree.py (but should be changed)
    names_to_check = []
    dist_check_list = []
    for name, tree_object in map_names_to_branches.items():
        if "bud" in name:
            if tree_object.branch_child != None:
                if tree_object.branch_child.growth.age_in_years == 0:
                    if name not in names_to_check:
                        dist_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)
            if tree_object.spur_child != None:
                    if name not in names_to_check:
                        dist_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)


    referance_object = dist_prune(tree_sim_base,
                                  branch_hierarchy=branch_hierarchy,
                                  map_names_to_branches=map_names_to_branches,
                                  check=dist_check_list)


    for branch in referance_object:
        row_exists = (df.values == [branch.start_loc.x, 
                                    branch.start_loc.y, 
                                    branch.start_loc.z, 
                                    RADIUS,
                                    branch.name]).all(axis=1).any()

        if not row_exists:
            new_row = pd.DataFrame([{'marked_x':branch.start_loc.x, 
                                     'marked_y':branch.start_loc.y, 
                                     'marked_z':branch.start_loc.z,
                                     'radius':RADIUS,
                                     'name':branch.name}])
            
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            print(f'[ERROR] Duplicate referance location')

    df.to_csv(Path(__file__).parent.resolve()/"marked_locations.csv", 
              index_label="index", 
              mode='w')

    return 
