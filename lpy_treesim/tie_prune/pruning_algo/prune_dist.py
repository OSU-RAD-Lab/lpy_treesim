from lpy_treesim.tie_prune.pruning_algo.ltr_and_data_structure import NeighborsDataStructure
        
def dist_prune(tree_sim_base,
               branch_hierarchy: dict, 
               map_names_to_branches: dict,
               mark: bool,
               radius: float = 0.05,
               print_to_terminal: bool = False):
    
    PRINT = print_to_terminal
    RADIUS = radius
    names_to_x = []
    pruned = []
    referance_object = []

    # TODO: update so that primary branches are not purned only primary branch buds
    names_to_check = []
    bud_check_list = []
    for name, tree_object in map_names_to_branches.items():
        if "bud" in name and "trunk" not in name:
            '''
            if tree_object.branch_child != None:
                if tree_object.branch_child.growth.age_in_years == 0:
                    if name not in names_to_check:
                        bud_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)
            '''
            if tree_object.spur_child != None:
                    if PRINT and tree_object.branch_child != None:
                        print(f'Branch Child {tree_object.branch_child.name}')
                        print(f'Spur Children {tree_object.spur_child.name}')
                    if name not in names_to_check:
                        bud_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)
                    
    
    def prune(key_name,
              query,
              names_to_x=names_to_x, 
              pruned=pruned,
              check=bud_check_list, 
              map_names_to_branches=map_names_to_branches):
        
        if not mark: names_to_x.extend(map_names_to_branches[key_name].prune())
        pruned.append(key_name)
        if PRINT: print(f'{key_name} pruned becuase of {query}')
        for test in check[:]:
            if test.name == key_name:
                check.remove(test)
    
    loop = True
    while loop and (len(bud_check_list)>0):

        index = None
        structure = bud_check_list+referance_object
        spur_buds_spatial = NeighborsDataStructure(None, structure) #Faster to rebuild??

        if len(referance_object) > 0:
            neighbors_total = {}
            for ref in referance_object:
                neighbors, number = spur_buds_spatial.query_objects_within_r(ref.name, 
                                                                             radius=(2*RADIUS))
                neighbors_total.update(neighbors)

            #unique_neighbors = list(set(neighbors_total))
            for n, test in enumerate(bud_check_list):
                if test.name not in list(neighbors_total.keys()):
                    index = n
                    break

            if index == None:
                if PRINT: print('no referance location found removing rest of check')
                for remaining in bud_check_list:
                    if not mark: names_to_x.extend(map_names_to_branches[remaining.name].prune())
                loop = False
                break

        else:
            index = 0

        referance_object.append(bud_check_list[index])
        query = bud_check_list[index].name
        neighbors, number = spur_buds_spatial.query_objects_within_r(query, radius=RADIUS)
        neighbors.pop(query, None)
        bud_check_list.pop(index)

        if number >= 1:
            for key_name in list(neighbors.keys()):
                if len(pruned) > 0:
                    if key_name not in pruned:
                        prune(key_name, query)
                    else:
                        print("[Error] Bud already removed")

                else:
                    prune(key_name, query)


    if mark:
        location = [location.start_loc for location in referance_object]
        LENGTH = len(location)
        type = ["bud_spacing"]*LENGTH
        radius = [RADIUS]*LENGTH
        normal_vector = [[(0,0,0),(0,0,0)]]*len(location)
        year = [int(tree_sim_base.current_iteration/tree_sim_base.config.num_iter_per_year)]*LENGTH
        name = [bud.name for bud in referance_object]*LENGTH


        return location, type, radius, normal_vector, year, name

    else:
        for name in names_to_x:
            del branch_hierarchy[name]
            del map_names_to_branches[name]

        return None, None, None, None, None, None