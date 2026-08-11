import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
import numpy as np
import time as tm
import pandas as pd
from pathlib import Path
# print = lambda *args, **kwargs: None

# Class to have use a dot dictionary data structure for the output
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value

class NeighborsDataStructure:
    """Build and query a spatial index for nearby tree branches.

    The class stores branch coordinates in a KD-tree so that nearby branches can
    be discovered efficiently for pruning or tie-related analyses.
    """

    def __init__(self, 
                 map_names_to_branches, 
                 to_index="bud" # can be name to index or a list of tree objects
                                # must have atribute .location.start or .start_loc
                 ):
        """Initialize the neighbor-search structure from branch objects.

        Args:
            map_names_to_branches: Mapping of branch names to branch object
                collections that should be indexed. Can be set to None if list is
                provided in to_index.
            to_index: Can be name to index, such as "bud" or a list of tree objects
                from map_names_to_branches or branch_hierarchy. If list of objects,
                the objects must have either .location.start or .start_loc atribute
                (can only have one, not both, becuase I am lazy).
        """
        
        self.map_names_to_branches = map_names_to_branches
        self.tree_objects = []
        self.query_made = False

        # TODO: code is not doing what it is supposed to but it works
        if type(to_index) is list:
            self.tree_objects = to_index
        elif type(to_index[0]) is str:
            for name, objects in self.map_names_to_branches.items():
                if to_index in name:
                    self.tree_objects.append(objects)
        else:
            raise ValueError('Unsported input for to_index')
            

        if type(to_index) is str:
            if "bud" in to_index: 
                self.coords = np.array([obj.start_loc for obj in self.tree_objects], 
                                        dtype=np.float64)
        elif type(to_index) is list:
            if "bud" in to_index[0].name:
                self.coords = np.array([obj.start_loc for obj in self.tree_objects], 
                                        dtype=np.float64)
        else:        
            self.coords = np.array([obj.location.start for obj in self.tree_objects], 
                                    dtype=np.float64)

        self.tree_names = np.array([obj.name for obj in self.tree_objects])                    
        if self.tree_names.size == 0:
            raise ValueError('Locations must be given')

        #create the data structure  
        self.tree = KDTree(self.coords)
    
    def query_objects_within_r(self, 
                               referance_name = 60, # Can be a name or index 
                               radius = 0.2,
                               output_type="dict"):
        """Find all indexed objects within a given radius of a reference branch.

        Args:
            referance_name: Name of the reference branch or its integer index.
            radius: Search radius in the same coordinate system as the stored
                branch locations.
            output_type: Output format; "dict" returns a nested mapping while
                other values return a structured NumPy array.

        Returns:
            A dictionary-like object or NumPy structured array containing the
            matching branch names, coordinates, and distances.
        """
        self.radius = radius
        self.query_made = True
        start_time = tm.perf_counter()
        if type(referance_name) is int:
            reference_index = referance_name
        else:
            reference_index = int(np.where(self.tree_names == referance_name)[0][0])
        
        self.query_points = self.coords[reference_index]
        self.neighbor_indices = np.asarray(self.tree.query_ball_point(self.query_points, 
                                                                      self.radius, 
                                                                      return_length=False), 
                                      dtype=np.int32)

        self.neighbor_points = self.coords[self.neighbor_indices]
        distances = np.linalg.norm(self.neighbor_points - self.query_points, axis=1)

        self.s_delta = tm.perf_counter() - start_time

        #print(f"{self.neighbor_indices.size - 1} Neighbors found using data structure in {s_delta} sec")

        if output_type == 'dict':
            output = {}
            for name, location, distance in zip(self.tree_names[self.neighbor_indices], self.neighbor_points, distances):
                output[name] = {'location':location.tolist(), 'distance':float(distance)}
                output = DotDict(output)
        elif output_type == 'indicies':
            output = self.neighbor_indices
        else:  
            # If you want the output to be a numpy array instead
            dtype = [('name'), 
                    ('location', np.float64, (3,)), 
                    ('distance', np.float64)]

            output = np.empty(self.tree_names[self.neighbor_indices].size, dtype=dtype)
            output['name'] = self.tree_names[self.neighbor_indices]
            output['location'] = self.neighbor_points
            output['distance'] = distances
        
        return output, (self.neighbor_indices.size-1)
    
    def find_edges(self):
        """Create line segments connecting the query point to each neighbor.

        Returns:
            A tuple of arrays containing the x, y, and z coordinates for each
            connecting edge.
        """
        line_query = np.tile(self.query_points.tolist(), 
                                (self.neighbor_points[:, 0].size, 1))
        
        x_line = np.concatenate((line_query[:, [0]], self.neighbor_points[:, [0]]), axis=1)
        y_line = np.concatenate((line_query[:, [1]], self.neighbor_points[:, [1]]), axis=1)
        z_line = np.concatenate((line_query[:, [2]], self.neighbor_points[:, [2]]), axis=1)

        return x_line, y_line, z_line
    
    def make_plot(self, save_to= "data/nn_plot.png"):
        """Create and save a 3D plot of the reference branch and its neighbors.

        Args:
            save_to: File path where the generated plot image should be written.

        Raises:
            ValueError: If a query has not been executed yet.
        """
        if self.query_made == False:
            raise ValueError("Query must be run to plot the referance and objects found")
        
        # Set up the 3D canvas
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        
        ax.scatter(self.neighbor_points[:, [0]], 
                   self.neighbor_points[:, [1]],
                   self.neighbor_points[:, [2]],
                   c='green', 
                   s=15, 
                   marker = "x")
        
        x_line, y_line, z_line = self.find_edges()

        ax.plot(x_line, y_line, z_line, c='red', alpha=0.2, lw = 0.6)

        ax.scatter(self.query_points[0], 
                   self.query_points[1], 
                   self.query_points[2], 
                   c='green', 
                   s=85, 
                   marker='.')

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Z Axis")
        ax.set_zlabel("Y Axis")
        
        legend_elements = [
            
            #Line2D([0], [0], 
            #    marker='.', 
            #    color='w',  
            #    markerfacecolor='blue', 
            #    markersize=15, 
            #    label='w/ Iteration'),
                
            Line2D([0], [0], 
                marker='x', 
                color='green', 
                linestyle='None', 
                markersize=8, 
                label='w/ Data Structure'),

            Line2D([0], [0], 
                marker='.', 
                color='w', 
                markerfacecolor='green', 
                markersize=15, 
                label='Referance Bud')
            
            ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.view_init(elev=60, azim=15, roll= 0)
        
        # Force an export of the image directly to the folder.
        # This acts as a backup in case the environment blocks the pop-up window.
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_to}")

    def mark_referance(self):
        """Export the query result as CSV files for downstream use.

        The method writes edge and node data describing the reference branch and
        its neighboring branches to the data directory.

        Raises:
            ValueError: If a query has not been executed yet.
        """
        if self.query_made == False:
            raise ValueError('Query must be run to mark the referance and objects found')
        # Added this code in order to show what's generated by the plot within the sim
        edge_data = []
        x_line, y_line, z_line = self.find_edges()

        for x, y, z in zip(x_line, y_line, z_line):
            edge_data.append({'x1': x[0], 
                              'y1': y[0],
                              'z1': z[0], 
                              'x2': x[1], 
                              'y2': y[1],
                              'z2': z[1]
                            })
        
        df_edges = pd.DataFrame(edge_data)
        df_edges.to_csv("data/knn_edges.csv", index=False)

        node_data = [{'x': self.query_points[0], 
                      'y': self.query_points[1], 
                      'z': self.query_points[2], 
                      'type': 'ref'
                      }]
        
        for locations in self.neighbor_points:
            node_data.append({'x': locations[0], 
                              'y': locations[1], 
                              'z': locations[2], 
                              'type': 'neighbor'
                              })
        
        df_nodes = pd.DataFrame(node_data)
        df_nodes.to_csv("data/knn_nodes.csv", index=False)
        # Added this, end of new code changes

    # function for testing purposes
    def test_time(self):
        """Benchmark the same radius query without using the KD-tree.

        Args:
            referance_name: Name of the reference branch or its integer index.
            radius: Search radius in the same coordinate system as the stored
                branch locations.

        Returns:
            Time difference of iteration.
        """
        if self.query_made != True:
            raise RuntimeError ("Query to data structure must be made first to compare")

        start_time = tm.perf_counter()

        neighbor_indices = []
        neighbor_points = []
        distances = []

        for index, point in enumerate(self.coords):
            distance = np.linalg.norm(point - self.query_points)
            if distance <= self.radius:
                neighbor_indices.append(index)
                neighbor_points.append(point)
                distances.append(distance)

        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
        neighbor_points = np.asarray(neighbor_points, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)

        i_delta = tm.perf_counter() - start_time
        print(f"{neighbor_indices.size} Neighbors found by iteration in {i_delta} sec")
        print(f"{self.neighbor_indices.size} Neighbors found by iteration in {self.s_delta} sec")
        
        return i_delta, self.s_delta

class LtrHuristic:
    def __init__(self, map_names_to_branches, tree_sim=None):
        self.map_names_to_branches = map_names_to_branches
        self.tree_sim = tree_sim

    def get_tcsa(self, height_m: float = 0.3) -> float:
        """
        Calculates the Trunk Cross-Sectional Area (TCSA) in cm² 
        at a specific height.
        """
        trunk = None
        for name, obj in self.map_names_to_branches.items():
            if "trunk" in name.lower():
                trunk = obj
                break

        if trunk is None:
            print("Error: No trunk found in map_names_to_branches.")
            return 0.0
            
        if trunk.growth.length < height_m:
            print(f"Trunk ({trunk.growth.length:.3f}m) has not reached {height_m}m.")
            return 0.0

        # t_ratio = height_m / trunk.growth.length
        
        diameter_at_height_m = trunk.growth.get_diameter(dist_along=height_m)
        
        # Extract the radius in centimeters
        radius_cm = (diameter_at_height_m / 2.0) * 100
        
        # Calculate Area: A = pi * r^2
        tcsa_cm2 = np.pi * (radius_cm ** 2)
        
        print(f"\nMetrics: TCSA at {height_m}m is {tcsa_cm2:.4f} cm²")
        return tcsa_cm2

    def get_primary_limbs(self) -> list:
        """
        Scans the tree dictionary, isolates all primary wood limbs
        and stores them in a list for the LCSA calculations.
        """
        primary_limbs = []
        kept_limbs = []
        csv_log_queue = []
        
        # 1. Scan every piece of wood/node on the tree
        for name, obj in self.map_names_to_branches.items():
            
            # 2. Filter for primary limbs, ensuring it has a growth attribute
            if "primary" in name.lower() and hasattr(obj, "growth"):
                primary_limbs.append((name, obj))
                
        # 3. Print the total count and the names of the limbs found
        print(f"Metrics: Found {len(primary_limbs)} primary limbs.")
        for limb_name, limb_obj in primary_limbs:
            print(f" -> {limb_name} (Length: {limb_obj.growth.length:.2f}m)")
            
        return primary_limbs
    
    def get_lcsa_metrics(self, primary_limbs: list, measurement_dist_m: float = 0.025) -> dict:
        """
        Calculates the Limb Cross-Sectional Area (LCSA) in cm² for primary limbs
        at a specific distance from the trunk. 
        """
        valid_limbs = []
        too_short_limbs = []
        
        for name, obj in primary_limbs:
            length_m = obj.growth.length
            
            if length_m < measurement_dist_m:
                too_short_limbs.append((name, obj))
                continue
                
            # t_ratio = measurement_dist_m / length_m
            
            diameter_m = obj.growth.get_diameter(dist_along=measurement_dist_m)
            
            # Extract the radius in centimeters
            radius_cm = (diameter_m / 2.0) * 100
            
            # Calculate Area: A = pi * r^2
            lcsa_cm2 = np.pi * (radius_cm ** 2)
            
            valid_limbs.append({
                "name": name,
                "object": obj,
                "lcsa_cm2": lcsa_cm2
            })
            
        print(f"\nMetrics: Limb LCSA AT {measurement_dist_m * 100}cm ")
        print(f"Metrics: Valid Limbs Calculated: {len(valid_limbs)}")
        for limb in valid_limbs:
            print(f"  -> {limb['name']}: LCSA = {limb['lcsa_cm2']:.2f} cm²")
            
        if len(too_short_limbs) > 0:
            print(f"\nLimbs Too Short (<{measurement_dist_m * 100}cm): {len(too_short_limbs)}")
            for name, obj in too_short_limbs:
                print(f"  -> {name} (Length: {obj.growth.length:.3f}m)")
        print("\n")
            
        return {"valid": valid_limbs, "too_short": too_short_limbs}
    def simulate_ltr_pruning(self, tcsa_cm2: float, limb_metrics: dict, target_ltr: float = 0.5) -> dict:
        """
        Calculates the initial LTR and iteratively removes the largest valid limbs
        until the target LTR threshold is achieved. Validates replacements via Energy Matrix.
        """
        if tcsa_cm2 <= 0:
            print("Error: Invalid TCSA. Cannot calculate LTR.")
            return {}

        valid_limbs = limb_metrics.get("valid", [])
        sorted_limbs = sorted(valid_limbs, key=lambda x: x['lcsa_cm2'], reverse=True)
        
        total_lcsa = sum(limb['lcsa_cm2'] for limb in sorted_limbs)
        current_ltr = total_lcsa / tcsa_cm2

        
        print(f"\nMetrics: Trees to be pruned")
        print(f"Initial Total LCSA: {total_lcsa:.2f} cm²")
        print(f"Baseline TCSA: {tcsa_cm2:.2f} cm²")
        print(f"Initial LTR: {current_ltr:.4f}")
        print(f"Target LTR: {target_ltr:.4f}\n")
        

        primary_to_prune = []
        kept_limbs = []
        csv_log_queue = []  # Tracks everything for the CSV
        
        while current_ltr > target_ltr and len(sorted_limbs) > 0:
            largest_limb = sorted_limbs.pop(0)
            limb_obj = largest_limb['object']
            branch_id = int(largest_limb['name'].split("_")[-1])

            is_viable_replacement = False
            replacement_name = None

            if self.tree_sim:
                # 1. Identify which wire the removable branch is currently tied to
                target_wire_id = None
                for w_id, wire in enumerate(self.tree_sim.branch_attractor):
                    if wire.branch_id == branch_id:
                        target_wire_id = w_id
                        break

                # 2. Gather all untied candidates (buds and untied branches)
                candidates = []
                for name, obj in self.map_names_to_branches.items():
                    # Skip the limb we are actively trying to remove
                    if obj == limb_obj:
                        continue
                        
                    is_bud = "bud" in name.lower()
                    is_untied_branch = hasattr(obj, "tying") and not obj.tying.is_tied
                    
                    if is_bud or is_untied_branch:
                        candidates.append(obj)

                # 3. Run the Energy Matrix with the removable branch's wire forced open
                energy_matrix, wire_ids, open_branches = self.tree_sim.get_energy_matrix(
                    branches=candidates, 
                    flagged_branch_ids=[branch_id]
                )
                
                # 4. Check the proposed assignments without actually tying them
                proposed_ties = self.tree_sim.test_assignment_viability(energy_matrix, wire_ids, open_branches)
                
                # 5. See if any candidate successfully map to a newly opened wire
                if target_wire_id is not None:
                    for cand_name, proposed_wire_id in proposed_ties.items():
                        if proposed_wire_id == target_wire_id:
                            is_viable_replacement = True
                            replacement_name = cand_name
                            break
                else:
                    # If the primary limb wasn't tied to a wire, we don't strictly need a wire replacement
                    is_viable_replacement = True

            # Dormancy Roll (Only roll if the chosen replacement is a bud) 
            if is_viable_replacement and replacement_name and "bud" in replacement_name.lower():
                import random
                dormancy_chance = getattr(self.tree_sim.config, 'dormancy_probability', 0.7)
                
                if random.random() > dormancy_chance:
                    print(f"  -> Dormancy Check: Bud {replacement_name} failed to wake up.")
                    is_viable_replacement = False # Force the fallback
                else:
                    print(f"  -> Dormancy Check: Bud {replacement_name} successfully broke dormancy!")

            # 6. Fallback Evaluation
            if not is_viable_replacement:
                print(f"Fallback Triggered: Limb {largest_limb['name']} lacks a viable tied replacement. Skipping.")
                kept_limbs.append(largest_limb) # Save the skipped limb

                # Log the skipped limb (Orange)
                csv_log_queue.append({
                    "limb_name": largest_limb['name'],
                    "type": "primary_without_replacement",
                    "object": limb_obj,
                    "replacement_name": None
                })
                continue
                
            largest_limb['renewal_candidate'] = replacement_name 
            primary_to_prune.append(largest_limb)
            
            # 1. Log the primary limb getting pruned (Red)
            csv_log_queue.append({
                "limb_name": largest_limb['name'],
                "type": "primary_to_prune",
                "object": limb_obj,
                "replacement_name": replacement_name
            })
            
            # 2. Log the replacement if one exists (Cyan)
            if replacement_name and replacement_name in self.map_names_to_branches:
                csv_log_queue.append({
                    "limb_name": replacement_name,
                    "type": "flag_for_replace",
                    "object": self.map_names_to_branches[replacement_name],
                    "replacement_name": None
                })

            total_lcsa -= largest_limb['lcsa_cm2']
            current_ltr = total_lcsa / tcsa_cm2
            
            replacement_str = f"Replaced by: {replacement_name}" if replacement_name else "No wire replacement needed"
            print(f"Limb Marked For Removal: {largest_limb['name']} ({replacement_str}) -> New LTR: {current_ltr:.4f}")
            
        print(f"\nFinal LTR Achieved: {current_ltr:.4f}")
        print(f"Total Limbs Flagged for Removal: {len(primary_to_prune)}")
        
        # Combine any remaining untouched limbs with the skipped limbs
        kept_limbs.extend(sorted_limbs)
        print(f"Total Limbs Kept: {len(kept_limbs)}")
        print("\n")

        
        # 7. CSV Logging for All Marked Items
        if csv_log_queue:
            import pandas as pd
            from pathlib import Path

            marker_data = []

            for i, log_item in enumerate(csv_log_queue):
                obj = log_item['object']
                name = log_item['limb_name']
                item_type = log_item['type']
                
                # 1. Base coordinates
                is_bud = hasattr(obj, 'start_loc')
                if is_bud:
                    base_x, base_y, base_z = obj.start_loc.x, obj.start_loc.y, obj.start_loc.z
                else:
                    base_x, base_y, base_z = obj.location.start.x, obj.location.start.y, obj.location.start.z
        
                # 2. Extract the normalized directional vector (fallback)
                if hasattr(obj, 'start_dir'):
                    dir_x, dir_y, dir_z = obj.start_dir.x, obj.start_dir.y, obj.start_dir.z
                else:
                    dir_x, dir_y, dir_z = obj.location.start_dir.x, obj.location.start_dir.y, obj.location.start_dir.z

                loc_x, loc_y, loc_z = base_x, base_y, base_z
                norm_x2, norm_y2, norm_z2 = loc_x + dir_x, loc_y + dir_y, loc_z + dir_z

                
                if item_type in ["primary_to_prune", "primary_without_replacement"]:
                    
                    # Find the first existing bud on this branch to establish the true curved vector. Stops marker from floating, kinda
                    bud_loc = None
                    for b_idx in range(10): # Check first 10 possible buds
                        bud_name = f"{name}_bud_{b_idx}"
                        if bud_name in self.map_names_to_branches:
                            bud_loc = self.map_names_to_branches[bud_name].start_loc
                            break
                    
                    if bud_loc:
                        bx, by, bz = bud_loc.x, bud_loc.y, bud_loc.z
                        
                        # Calculate physical distance to the first bud
                        dist_to_bud = ((bx - base_x)**2 + (by - base_y)**2 + (bz - base_z)**2) ** 0.5
                        
                        if dist_to_bud > 0.001:
                            # Move 10cm out to clear the  trunk.
                            # Cap at 90% of the distance to the first bud so it doesn't overshoot.
                            fraction = min(0.10 / dist_to_bud, 0.9)
                            
                            loc_x = base_x + (bx - base_x) * fraction
                            loc_y = base_y + (by - base_y) * fraction
                            loc_z = base_z + (bz - base_z) * fraction
                        
                        # Update normal vector to aim directly at the first bud (perfect tilt)
                        norm_x2, norm_y2, norm_z2 = bx, by, bz
                    else:
                        # Fallback if no bud exists: use a larger linear offset (10cm)
                        offset_dist = 0.10 
                        loc_x = base_x + (dir_x * offset_dist)
                        loc_y = base_y + (dir_y * offset_dist)
                        loc_z = base_z + (dir_z * offset_dist)
                        norm_x2 = loc_x + dir_x
                        norm_y2 = loc_y + dir_y
                        norm_z2 = loc_z + dir_z
        
                
                norm_x1, norm_y1, norm_z1 = loc_x, loc_y, loc_z

                # Safely extract radius (buds do not have growth attributes)
                if hasattr(obj, 'growth'):
                    radius = obj.growth.get_diameter(0.0) / 2.0
                else:
                    radius = 0.005 # Fallback radius for buds
                

                # Calculate the exact year integer to match the .obj export suffix
                current_year = 0
                if getattr(self, 'tree_sim', None) and getattr(self.tree_sim, 'config', None):
                    iters_per_year = self.tree_sim.config.num_iter_per_year
                    if iters_per_year > 0:
                        current_year = int((self.tree_sim.current_iteration / iters_per_year) + 1)

                marker_data.append({
                    'index': i,
                    'Loc_x': loc_x,
                    'Loc_y': loc_y,
                    'Loc_z': loc_z,
                    'Type': item_type,
                    'Radius': radius,
                    'Norm_x1': norm_x1,
                    'Norm_y1': norm_y1,
                    'Norm_z1': norm_z1,
                    'Norm_x2': norm_x2,
                    'Norm_y2': norm_y2,
                    'Norm_z2': norm_z2,
                    'Year': current_year,
                    'Name': name
                })

            df = pd.DataFrame(marker_data)
            csv_path = Path(__file__).parent.resolve() / "marked_locations.csv"

            # Append to existing file or create a new one with headers
            if csv_path.exists():
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)
        
        return {
            "kept_limbs": kept_limbs, 
            "primary_to_prune": primary_to_prune,
            "final_ltr": current_ltr
        }