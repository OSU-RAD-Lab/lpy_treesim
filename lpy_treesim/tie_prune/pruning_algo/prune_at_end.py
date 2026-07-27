from lpy_treesim.tie_prune.pruning_algo.radius_of_neighbors_class import NeighborsDataStructure
import pandas as pd
from pathlib import Path


#Set true for dignostic info (Errors still printed when False)
PRINT = False
RADIUS = 0.1

# changed from (self, branch_hierarchy: dict, map_names_to_branches: dict):
def end_prune(tree_sim_base, 
              branch_hierarchy: dict, 
              map_names_to_branches: dict):
    
    df = pd.DataFrame(columns=['marked_x', 'marked_y', 'marked_z', 'radius', 'name'])
    if PRINT: print("[end_prune] Function Called, end_prune() running...")
   
    names_to_x = []
    check = []
    pruned = []
    names_to_check = []
    referance_object = []

    def prune(key_name,
              query,
              names_to_x=names_to_x, 
              pruned=pruned,
              check=check, 
              map_names_to_branches=map_names_to_branches):
        
        names_to_x.extend(map_names_to_branches[key_name].prune())
        pruned.append(key_name)
        print(f'{key_name} pruned becuase of {query}')
        for test in check[:]:
            if test.name == key_name:
                check.remove(test)
    
    for name, tree_object in map_names_to_branches.items():
        if "bud" in name:
            if tree_object.branch_child != None:
                if tree_object.branch_child.growth.age_in_years == 0:
                    if name not in names_to_check:
                        check.append(tree_object)
                        names_to_check.append(tree_object.name)
                    else:
                        print('already in check list')
            if tree_object.spur_child != None:
                    if name not in names_to_check:
                        check.append(tree_object)
                        names_to_check.append(tree_object.name)
                    else:
                        print('already in check list')
   
    loop = True
    while loop and (len(check)>0):

        index = None
        structure = check+referance_object
        spur_buds_spatial = NeighborsDataStructure(None, structure) #Faster to rebuild??

        if len(referance_object) > 0:
            neighbors_total = {}
            for ref in referance_object:
                neighbors, number = spur_buds_spatial.query_objects_within_r(ref.name, 
                                                                            radius=(2*RADIUS))
                neighbors_total.update(neighbors)

            #unique_neighbors = list(set(neighbors_total))
            for n, test in enumerate(check):
                if test.name not in list(neighbors_total.keys()):
                    index = n
                    break

            if index == None:
                print('no referance location found removing rest of check')
                for remaining in check:
                    names_to_x.extend(map_names_to_branches[remaining.name].prune())
                loop = False
                break

        else:
            index = 0

        referance_object.append(check[index])
        query = check[index].name
        neighbors, number = spur_buds_spatial.query_objects_within_r(query, radius=RADIUS)
        neighbors.pop(query, None)
        check.pop(index)

        if number >= 1:
            for key_name in list(neighbors.keys()):
                if len(pruned) > 0:
                    if name not in pruned:
                        prune(key_name, query)
                    else:
                        print("[Error] Bud already removed")

                else:
                    prune(key_name, query)
    
    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]

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

    df.to_csv(Path(__file__).parent.resolve()/"marked_locations.csv", index_label="index", mode='w')

    return 
