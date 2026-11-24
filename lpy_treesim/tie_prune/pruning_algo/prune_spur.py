from lpy_treesim.tie_prune.pruning_algo.ltr_and_data_structure import NeighborsDataStructure
from lpy_treesim.tie_prune.pruning_algo.mark_cuts import mark_cuts
        
def spur_prune(tree_sim_base,
               branch_hierarchy: dict, 
               map_names_to_branches: dict,
               mark: bool,
               radius: float = 0.05,
               print_to_terminal: bool = False):
    """Prune competing spur buds based on local spacing constraints.

    The algorithm picks a reference spur, finds nearby spur buds within a
    tolerance radius, and removes the neighbors that are too close. When
    ``mark`` is enabled the prune locations are returned as an entry to 
    the marked_locations.csv file instead of deleting branches from the 
    hierarchy which will cause the branch to not be rendered.
    """

    PRINT = print_to_terminal

    # Minimum allowed spacing between neighboring spur buds, a spur is considered
    # too close to another spur when its nearest neighbor is within radius.
    RADIUS = radius

    # Tracks branches removed during pruning so they can be deleted from the
    # branch hierarchy.
    names_to_x = []
    pruned = []

    # Remaining reference buds used to decide which local cluster of spur buds is
    # the canonical one to keep. A bud is added here once it is selected as the
    # anchor around which nearby competitors are evaluated.
    referance_object = []

    # When true, the code actually prunes the marked buds as part of the current
    # year's growth cycle. Setting this to false preserves the marking-only mode
    # used for analysis or visualization.
    PRUNE_MARKS = True
    # TODO: PRUNE_MARKS = True is currently causing "could not find" errors?
    # Now only "Bad growth length" errors?

    names_to_check = []
    bud_check_list = []

    # Find all buds that have spur children
    for name, tree_object in map_names_to_branches.items():
        if "bud" in name and "trunk" not in name:

            # If you want to check if a bud has a branch based on year rather than if
            # it has children.
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

        # Remove all spurs that are on the trunk
        if "bud" in name and "trunk" in name:
            if tree_object.spur_child != None:
                names_to_x.extend(tree_object.prune())
                

    def do_prune(key_name,
                 query,
                 names_to_x=names_to_x, 
                 pruned=pruned,
                 check=bud_check_list, 
                 map_names_to_branches=map_names_to_branches):
        """Prune a target spur and remove it from the candidate list.

        The function is used inside the local spacing loop to keep the working set
        of buds in sync with the actual tree state after a branch is removed.
        """
        names_to_x.extend(map_names_to_branches[key_name].prune())
        pruned.append(key_name)
        if PRINT: print(f'{key_name} pruned becuase of {query}')
        for test in check[:]:
            if test.name == key_name:
                check.remove(test)

    # Greedily select one spur as a reference site and prune all neighbors within
    # the allowed radius. Continue until no more candidate buds remain or we reach
    # a no-reference condition.
    loop = True
    while loop and (len(bud_check_list)>0):

        index = None
        # Build a neighbor lookup over the current active buds plus already chosen
        # reference objects so we can find remaining buds that are
        # close to a known referance bud.
        structure = bud_check_list + referance_object
        spur_buds_spatial = NeighborsDataStructure(None, structure)  # Faster to rebuild??

        if len(referance_object) > 0:
            neighbors_total = {}
            for ref in referance_object:
                neighbors, number = spur_buds_spatial.query_objects_within_r(ref.name, 
                                                                             radius=(2*RADIUS))
                neighbors_total.update(neighbors)

            # If a candidate bud is not within any existing reference neighborhood,
            # it becomes the next anchor point for a new pruning cluster.
            for n, test in enumerate(bud_check_list):
                if test.name not in list(neighbors_total.keys()):
                    index = n
                    break

            if index == None:
                if PRINT: print('no referance location found removing rest of check')
                if not mark or PRUNE_MARKS:
                    for remaining in bud_check_list:
                        names_to_x.extend(map_names_to_branches[remaining.name].prune())
                loop = False
                break

        else:
            # The first cluster always starts from the earliest bud in the list.
            # TODO: change this to be a random selection.
            index = 0

        referance_object.append(bud_check_list[index])
        query = bud_check_list[index].name
        neighbors, number = spur_buds_spatial.query_objects_within_r(query, radius=RADIUS)
        neighbors.pop(query, None)
        bud_check_list.pop(index)

        if not mark or PRUNE_MARKS:
            if number >= 1:
                for key_name in list(neighbors.keys()):
                    if len(pruned) > 0:
                        if key_name not in pruned:
                            do_prune(key_name, query)
                        else:
                            print("[Error] Bud already removed")
                    else:
                        do_prune(key_name, query)

    # Prune the remaining referance spur so the spurs do not get too long.
    if len(referance_object) > 0:
        for bud in referance_object:
            try:
                if map_names_to_branches[bud.spur_child.name].growth.length > 0.13:
                    # Stand-in cut: trim the spur to leave a single surviving bud.
                    map_names_to_branches[bud.spur_child.name].prune(0.1)
            except Exception:
                print("Not really sure why this is happening")

    if mark:
        # Return the marked pruning locations in the format expected by the rest of
        # the simulation pipeline so they can be visualized or logged later.
        location = [location.start_loc for location in referance_object]
        LENGTH = len(location)
        type = ["bud_spacing"]*LENGTH
        radius = [RADIUS]*LENGTH
        normal_vector = [[(0,0,0),(0,0,0)]]*LENGTH
        year = [int(tree_sim_base.current_iteration/tree_sim_base.config.num_iter_per_year)]*LENGTH
        name = [bud.name for bud in referance_object]

        if len(names_to_x)>0:
            for named in names_to_x:
                del branch_hierarchy[named]
                del map_names_to_branches[named]

        # Information that is written to a csv in prune_tree.py. This CSV is then read after
        # the tree is generated to add in the marking information.
        mark_cuts(location, type, radius, normal_vector, year, name) # imported function


    else:
        # In non-mark mode, delete all removed branches immediately from the active
        # hierarchy and branch-name index, and return placeholders.
        for named in names_to_x:
            del branch_hierarchy[named]
            del map_names_to_branches[named]
