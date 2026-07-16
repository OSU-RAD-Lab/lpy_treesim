from math import sqrt
from itertools import combinations as combo

def meters_to_inches(meters):
    return round(meters/39.3701, 4)

def prune_random(tree_sim_base, branch_hierarchy: dict, map_names_to_branches: dict):
    names_to_x = []
    check_spur = []
    check_side = []
    prune_locations = []
    for name, branch in map_names_to_branches.items():
        if "Spur" in name and "bud" not in name:
                check_spur.append(branch)
        if "Side" in name and "bud" not in name: #Tertiary for envy and Side  for ufo
                check_side.append(branch)
    
   