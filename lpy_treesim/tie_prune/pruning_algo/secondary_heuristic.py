
def secondary_heuristic_prune(tree_sim_base, 
                              branch_hierarchy:dict, 
                              map_names_to_branches:dict):
    marks = {'location':[], 'type':[], 'radius':[], 'normal_vector':[], 'year':[], 'name':[]}

    names_to_x = []
    end_branch_buds = []
    spur_buds = []

    for name, tree_object in map_names_to_branches.items(): 
        if "bud" in name and tree_object.branch_child != None:
            if "Primary" not in tree_object.branch_child.name:
                if len(branch_hierarchy[tree_object.branch_child.name]) == 0:
                    if name not in [n.name for n in end_branch_buds]:
                        end_branch_buds.append(tree_object)
                    else:
                        print('[Error] Already in end_branch_buds list')

    organized_tertiary_branches = {}
    buds = []
    branches = []
    n = 0
    for end_bud in end_branch_buds:
        buds.append(end_bud)
        check_bud = end_bud
        while "Primary" not in check_bud.wood_parent.name:
            buds.append(check_bud)
            branches.append(check_bud.wood_parent)

        reversed_buds = buds.reverse()
        organized_tertiary_branches[f'TertiaryBranch_{n}'] = {'buds':reversed_buds, 'branches':reversed_buds[1:]}
        n += 1

    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]
    return marks['location'], marks['type'], marks['radius'], marks['normal_vector'], marks['year'], marks['name']

    for name, tree_object in map_names_to_branches.items():
        if "Primary" in name and "bud" in name:
            if tree_object.branch_child != None:
                if name not in [n.name for n in tertiary_branch_buds]: 
                        tertiary_branch_buds.append(tree_object)
                else:
                    print('[Error] Already in tertiary_branch_buds list')
            if tree_object.spur_child != None:
                if name not in [n.name for n in spur_buds]:
                    spur_buds.append(tree_object)
                else:
                    print('[Error] Alread in spur_buds list')

        while len(tertiary_branch_buds)>0:
            for bud in tertiary_branch_buds:
                for child_bud in branch_hierarchy[bud.branch_child]:
                    print(child_bud)