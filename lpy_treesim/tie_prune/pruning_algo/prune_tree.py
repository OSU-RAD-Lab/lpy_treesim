from lpy_treesim.tie_prune.pruning_algo.prune_length import prune_length
from lpy_treesim.tie_prune.pruning_algo.prune_primary import prune_primary
from lpy_treesim.tie_prune.pruning_algo.prune_dist import dist_prune


def prune_tree(tree_sim_base, branch_hierarchy: dict, 
               map_names_to_branches: dict):
    
    # Take out old primary branches
    prune_primary(tree_sim_base=tree_sim_base, 
                  branch_hierarchy=branch_hierarchy, 
                  map_names_to_branches=map_names_to_branches)

    # Cut short any overly long branches
    prune_length(tree_sim_base=tree_sim_base, 
                 branch_hierarchy=branch_hierarchy, 
                 map_names_to_branches=map_names_to_branches)

    
    # Prunes everthing in a set radius based of a referance tree object 
    # Currently pruning primary branches which it should not do
    # dist_prune(tree_sim_base=tree_sim_base, 
    #            branch_hierarchy=branch_hierarchy, 
    #            map_names_to_branches=map_names_to_branches)

    # Function to mark to mark cuts, location should be outputed by functions such as prune_dist()

    def mark_cuts():
        pass

    '''
    df = pd.DataFrame(columns=['marked_x', 'marked_y', 'marked_z', 'radius', 'name'])
        if PRINT: print("[end_prune] Function Called, end_prune() running...")

        for branch in referance_object:
            row_exists = (df.values == [branch.start_loc.x, 
                                        branch.start_loc.y, 
                                        branch.start_loc.z, 
                                        RADIUS,
                                        branch.name]).all(axis=1).any()

            if not row_exists:
                new_row = pd.DataFrame([{'marked_x':branch.start_loc.x, 
                                        'marked_y':branch.start_loc.y, 
                                        'marked_z':branch.start_loc.z,
                                        'radius':RADIUS,
                                        'name':branch.name}])
                
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                print(f'[ERROR] Duplicate referance location')

        df.to_csv(Path(__file__).parent.resolve()/"marked_locations.csv", 
                index_label="index", 
                mode='w')

    return 
    '''