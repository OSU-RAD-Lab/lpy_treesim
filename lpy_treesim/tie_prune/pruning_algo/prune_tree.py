from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.primary_heuristic import primary_heuristic_prune
from lpy_treesim.tie_prune.pruning_algo.secondary_heuristic import secondary_heuristic_prune
from lpy_treesim.tie_prune.pruning_algo.prune_spur import spur_prune
import pandas as pd
from pathlib import Path

# Function to mark to mark cuts, location should be outputed by functions such as prune_dist()
    # This should be called after the ltr huristics cuts are marked becuase this fucntion will not
    # clear the csv file.

def mark_cuts(location, type, radius, normal_vector, year, name):    
    df = pd.read_csv(str(Path(__file__).parent.resolve() / "marked_locations.csv"))
    for location, type, radius, normal_vector, year, name in zip(location, type, radius, normal_vector, year, name):
        mask = (
                (df['Loc_x'] == location[0]) &
                (df['Loc_y'] == location[1]) &
                (df['Loc_z'] == location[2]) &
                (df['Type'] == type) 
                )
        row_exists = mask.any()

        if not row_exists:
            new_row = pd.DataFrame([{
                                        'Loc_x':location[0], 
                                        'Loc_y':location[1], 
                                        'Loc_z':location[2],
                                        'Type':type,
                                        'Radius':radius,
                                        'Norm_x1':normal_vector[0][0],
                                        'Norm_y1':normal_vector[0][1],
                                        'Norm_z1':normal_vector[0][2],
                                        'Norm_x2':normal_vector[1][0],
                                        'Norm_y2':normal_vector[1][1],
                                        'Norm_z2':normal_vector[1][2],
                                        'Year':year,
                                        'Name':name}])
            
            df = pd.concat([new_row], ignore_index=False)
        else:
            print(f'[ERROR] Duplicate mark location')

        df.to_csv(Path(__file__).parent.resolve()/"marked_locations.csv", 
                mode='a',
                header=False,
                index=False)

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
    prune_length(tree_sim_base,
                 branch_hierarchy=branch_hierarchy, 
                 map_names_to_branches=map_names_to_branches,
                 mark=mark)

    
    # Prunes everthing in a set radius based of a referance tree object 
    # Currently pruning primary branches which it should not do
    location, type, radius, normal_vector, year, name = spur_prune(tree_sim_base=tree_sim_base, 
                                                                   branch_hierarchy=branch_hierarchy,
                                                                   map_names_to_branches=map_names_to_branches,
                                                                   mark = mark)
    
    if mark: mark_cuts(location, type, radius, normal_vector, year, name)
    

    # Placeholder fucntion
    location, type, radius, normal_vector, year, name = primary_heuristic_prune()
    
    # Funciton not working yet
    #location, type, radius, normal_vector, year, name = secondary_heuristic_prune(tree_sim_base=tree_sim_base, 
    #                                                                              branch_hierarchy=branch_hierarchy, 
    #                                                                              map_names_to_branches=map_names_to_branches)

    #mark_cuts(location, type, radius, normal_vector, year, name)