from lpy_treesim.tie_prune.pruning_algo.radius_of_neighbors_class import NeighborsDataStructure
        
def dist_prune(tree_sim_base,
               branch_hierarchy: dict, 
               map_names_to_branches: dict,
               check: list,
               radius=0.1):
    PRINT = False
    RADIUS = radius
    names_to_x = []
    pruned = []
    referance_object = []
    
    def prune(key_name,
                query,
                names_to_x=names_to_x, 
                pruned=pruned,
                check=check, 
                map_names_to_branches=map_names_to_branches):
        
        names_to_x.extend(map_names_to_branches[key_name].prune())
        pruned.append(key_name)
        if PRINT: print(f'{key_name} pruned becuase of {query}')
        for test in check[:]:
            if test.name == key_name:
                check.remove(test)
    
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
                if PRINT: print('no referance location found removing rest of check')
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
                    if key_name not in pruned:
                        prune(key_name, query)
                    else:
                        print("[Error] Bud already removed")

                else:
                    prune(key_name, query)
    
    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]

    return referance_object