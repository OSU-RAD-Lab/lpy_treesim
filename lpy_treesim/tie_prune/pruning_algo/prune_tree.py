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

    '''
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
'''
            # --- CSV export for Spheres, runs after all pruning 
            ltr = LtrHuristic(map_names_to_branches=map_names_to_branches)
            
            # 1. Get TCSA baseline
            tcsa = ltr.get_tcsa(height_m=0.3)
            
            # 2. Get the list of primary limbs
            primary_limbs = ltr.get_primary_limbs()
            
            # 3. Calculate LCSA metrics
            limb_metrics = ltr.get_lcsa_metrics(primary_limbs=primary_limbs, measurement_dist_m=0.025)
            
            # 4. Run the simulated LTR logic (purely mathematical, no pruning)
            ltr_results = ltr.simulate_ltr_pruning(tcsa_cm2=tcsa, limb_metrics=limb_metrics, target_ltr=0.5)

            removed_limbs = ltr_results.get("removed_limbs", [])
  
            sphere_data = []
            SPHERE_RADIUS = 0.035
                
            if removed_limbs:
                print(f"\nExporting LTR Cuts for CSV (Iteration {self.current_iteration})")
                removed_branch_names = [limb["name"] for limb in removed_limbs]
                    
                for branch_name, branch_children in branch_hierarchy.items():
                    if "trunk" not in branch_name:
                        continue
                            
                    for bud in branch_children:
                        if "bud" not in bud.name or not bud.branch_child:
                            continue
                                
                        if bud.branch_child.name in removed_branch_names:
                            print(f"Logging limb for sphere: {bud.branch_child.name}")
                            loc = bud.branch_child.location.start
                            sphere_data.append({
                                'iteration': self.current_iteration, # Adds the year/iteration 
                                'marked_x': loc.x, 
                                'marked_y': loc.y, 
                                'marked_z': loc.z,
                                'radius': SPHERE_RADIUS,
                                'name': bud.branch_child.name
                                })
                
                # Always export, even if empty, to prevent crashes and keep the timeline intact
                df = DF(sphere_data, columns=['iteration', 'marked_x', 'marked_y', 'marked_z', 'radius', 'name'])
                # csv_path = Path(__file__).parent.resolve() / "pruning_algo" / "marked_locations.csv"
                csv_path = Path(__file__).parent.resolve() / "pruning_algo" / "ltr_marked_locations.csv"
                
                # Wipe the file clean on the first pruning event , then append for all future years
                write_mode = 'w' if self.current_iteration < 30 else 'a'
                df.to_csv(csv_path, index_label="index", mode=write_mode, header=(write_mode == 'w'))
                
                if sphere_data:
                    print(f"Exported {len(sphere_data)} locations to {csv_path} (Mode: {write_mode})")
                print("\n")
            '''
