from itertools import combinations as combo
from math import sqrt

def get_dist(L1, L2):
    return sqrt((L1.x-L2.x)**2+(L1.y-L2.y)**2+(L1.z-L2.z)**2)

def prune_dist(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    names_to_x = []
    check_spur = []
    check_side = []
    for name, branch in map_names_to_branches.items():
        if "Spur" in name and "bud" not in name:
                check_spur.append(branch)
        if "Side" in name and "bud" not in name:
                check_side.append(branch)
    
   # for list in (check_spur, check_side):
    for referance_branch, comparison_branch in combo(check_side, 2):
        dist = get_dist(referance_branch.location.start, comparison_branch.location.start)
        if dist < 0.2:
            names_to_x.extend(comparison_branch.prune(at_length=0.001))
            #print(f"[DEBUG] Removing: {comparison_branch.name} becuase of {referance_branch.name} at {dist}")
    
    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]


'''
if branch.growth.length > 0.01:
    print(f"Pruning Branch {name} at {branch.growth.length}m to 0.0m")
    names_to_x.extend(branch.prune(at_length=0.01))
'''