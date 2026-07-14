from itertools import combinations as combo
from math import sqrt

#Set true for dignostic info (Errors still printed when False)
PRINT = True
RUN_ITERATIONS = False

def get_dist(L1, L2):
    return abs(sqrt((L1.x-L2.x)**2+(L1.y-L2.y)**2+(L1.z-L2.z)**2))

def end_prune(self, branch_hierarchy: dict, map_names_to_branches: dict):
    if PRINT: print("[end_prune] Function Called, end_prune() running...")
    branches = []
    buds = []

    names_to_x = []
    prune_locations = []
    
    for name, branch in map_names_to_branches.items():
        if "bud" in name:
            if branch.branch_child == None:
                buds.append(branch)
        if "bud" not in name:
            if "Side" in name: # "Tertiary" for envy and "Side" for ufo
                branches.append(branch)
            if "Spur" in name:
                branches.append(branch)
            if "Primary" in name:
                print("exists")
                branches.append(branch)
    '''
    for bud in buds:
        names_to_x.extend(bud.prune())
    '''

    RADIUS = 0.4 #distance in meters
    removed = []
    referance_branches = []
    index = 0

    while len(branches) != 0:
        if len(referance_branches) > 0:
            
            if not RUN_ITERATIONS:
                break
            
            end = False
            for n in range(0,len(branches)-1):
                dist1 = get_dist(branches[n].location.start, referance_branches[len(referance_branches)-1].location.start)     
                #print(dist1)
                if dist1 > (RADIUS*2):
                    if len(referance_branches) > 1:
                        for ref in referance_branches[:-1]:
                            dist2 = get_dist(branches[n].location.start, ref.location.start)
                            #print(dist2)
                            if dist2 > (RADIUS*2):
                                index = n
                                print(f'[DEBUG] DIST TO NEW REFERANCE: {round(dist, 4)} ({branches[index].name})')
                                end = True
                                break
                    else:
                        index = n
                        end = True
                        break   
                if end == True:
                    break

            if end == False:
                print(f'[ERROR] No referance branch found.')
                break
                
            referance = branches.pop(index)
            referance_branches.append(referance)

            #Debuging
            dist = get_dist(referance.location.start, referance_branches[len(referance_branches)-1].location.start)
            if dist < RADIUS:
                print(f"[ERROR] NEW REFERANCE TOO CLOSE. Last Referance: {referance_branches[len(referance_branches)-1].name}")
        else:
            referance = branches.pop(30)
            referance_branches.append(referance)
        
        if PRINT: print(f'[DEBUG] Referance: {referance.name}')
        #print(len(branches))
        comparisons = branches
        for compare in comparisons:
            
            #Debugging
            if referance in comparisons:
                print("[ERROR] referance should be removed")

            dist = get_dist(referance.location.start, compare.location.start)
            if dist < RADIUS:
                removed.append((compare.name, dist))
                names_to_x.extend(compare.prune(0.001))
                '''
                keep_list = []
                for child in branch_hierarchy[name]:
                    if child.name not in names_to_x:
                        keep_list.append(child)
                branch_hierarchy[name] = keep_list
                '''
                branches.remove(compare)
                #print(len(branches))
                #names_to_x.extend(referance.prune(0.001))

        '''
        for bud in buds:
            dist = get_dist(referance.location.start, bud.start_loc)
            if dist < RADIUS:
                removed.append((bud.name, dist))
                names_to_x.extend(bud.prune())
        '''

        if PRINT:
            print(f"{len(removed)} neighbors of referance {referance.name} removed")
            print(f"Referance at {referance.location.start}")
            print(f"    Branches removed:")

            for branch in removed:
                print(f"        {branch[0]:<17}| {round(branch[1], 2):.3f} m away")
            
            print(f"\n\n{len(referance_branches)} referance branch(es) used\n\n")
        removed = []

    for branch in referance_branches:
        if branch.location.start not in prune_locations:
            prune_locations.append(branch.location.start)
        else:
            print(f'[ERROR] Duplicate referance location')

    map_names_to_branches['marked_locations'] = prune_locations
    map_names_to_branches['radius'] = RADIUS - 0.005

    for name in names_to_x:
        del branch_hierarchy[name]
        del map_names_to_branches[name]
