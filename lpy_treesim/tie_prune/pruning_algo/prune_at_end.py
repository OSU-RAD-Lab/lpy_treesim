from itertools import combinations as combo
from math import sqrt

def get_dist(L1, L2):
    return sqrt((L1.x-L2.x)**2+(L1.y-L2.y)**2+(L1.z-L2.z)**2)

def end_prune(self, branch_hierarchy: dict, map_names_to_branches: dict):
    bud_list = []
    names_to_x = []
    prune_locations = []
    for name, branch in map_names_to_branches.items():
        if "bud" in name:
            bud_list.append(branch)
    
    referance = None
    for referance_bud, comparison_bud in combo(bud_list,2):
        if referance != referance_bud:
            within_list = []
            referance = referance_bud
        dist = get_dist(referance_bud.start_loc, comparison_bud.start_loc)
        if dist < 0.1:
            #names_to_x.extend(comparison_bud.prune())
            prune_locations.append(referance_bud.start_loc)
            #print(f"[DEBUG] {comparison_branch.name} becuase of {referance_branch.name} at {dist}")
    
    map_names_to_branches['marked_locations'] = prune_locations

    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]