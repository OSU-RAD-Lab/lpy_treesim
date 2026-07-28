#from lpy_treesim.tie_prune.tie_prune_simulation_base import TreeSimulationBase
from math import sqrt
from itertools import combinations as combo

def meters_to_inches(meters):
    return round(meters/39.3701, 4)

def get_dist(L1, L2):
    return sqrt((L1.x-L2.x)**2+(L1.y-L2.y)**2+(L1.z-L2.z)**2)

def prune_random(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    names_to_x = []
    check = []
    for name, branch in map_names_to_branches.items():
        if "Spur" in name or "Side" in name:
            if "bud" not in name:
                check.append(branch)

    for referance_branch, comparison_branch in combo(check, 2):
        dist = get_dist(referance_branch.location.start, comparison_branch.location.start)
        if dist < 100:
            #print(f"[DEBUG] dist of obj removed: {dist}")
            names_to_x.extend(comparison_branch.prune(at_length=0.001))
    
    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]



'''
if branch.growth.length > 0.01:
    print(f"Pruning Branch {name} at {branch.growth.length}m to 0.0m")
    names_to_x.extend(branch.prune(at_length=0.01))
'''