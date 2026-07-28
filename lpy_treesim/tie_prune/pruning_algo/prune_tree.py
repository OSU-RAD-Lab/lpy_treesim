from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_dist import dist_prune

def prune_tree(tree_sim_base, branch_hierarchy: dict, 
               map_names_to_branches: dict):
    
    # Take out old primary branches
    prune_primary(tree_sim_base=tree_sim_base, branch_hierarchy=branch_hierarchy, map_names_to_branches=map_names_to_branches)

    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, 
                 branch_hierarchy=branch_hierarchy, 
                 map_names_to_branches=map_names_to_branches)

    # Prunes everthing in a set radius based of a referance tree object 
    names_to_check = []
    bud_check_list = []
    for name, tree_object in map_names_to_branches.items():
        if "bud" in name:
            if tree_object.branch_child != None:
                if tree_object.branch_child.growth.age_in_years == 0:
                    if name not in names_to_check:
                        bud_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)
            if tree_object.spur_child != None:
                    if name not in names_to_check:
                        bud_check_list.append(tree_object)
                        names_to_check.append(tree_object.name)

    dist_prune(tree_sim_base=tree_sim_base, 
               branch_hierarchy=branch_hierarchy, 
               map_names_to_branches=map_names_to_branches,
               check=bud_check_list)

    '''
    ltr = LtrHuristic(map_names_to_branches=map_names_to_branches)
    # Added this change
    # 1. Get TCSA baseline 
    tcsa = ltr.get_tcsa(height_m=0.3)
    
    # 2. Get the list of primary limbs
    primary_limbs = ltr.get_primary_limbs()
    
    # 3. Calculate LCSA metrics 
    limb_metrics = ltr.get_lcsa_metrics(primary_limbs=primary_limbs, measurement_dist_m=0.025)
    # Quick Prune Remove Later 
    # valid_limbs_to_prune = limb_metrics.get("valid", [])
    # valid_branch_names = [limb["name"] for limb in valid_limbs_to_prune]
    
    # names_to_x = []
    # for branch_name, branch_children in branch_hierarchy.items():
    #     if "trunk" not in branch_name:
    #         continue
            
    #     for bud in branch_children:
    #         if "bud" not in bud.name or not bud.branch_child:
    #             continue
    #         if bud.branch_child.name in valid_branch_names:
    #             print(f"Pruning valid limb: {bud.branch_child.name}")
    #             names_to_x.extend(bud.prune())
    # for name in names_to_x:
    #     if name in branch_hierarchy:
    #         del branch_hierarchy[name]
    #     if name in map_names_to_branches:
    #         del map_names_to_branches[name]
    '''
