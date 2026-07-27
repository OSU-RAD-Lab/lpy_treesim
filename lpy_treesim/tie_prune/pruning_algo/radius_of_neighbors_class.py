import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
import numpy as np
from numpy.dtypes import StringDType
import time as tm
import pandas as pd

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
    def __init__(self, map_names_to_branches:dict, to_index="bud"):
        
        self.map_names_to_branches = map_names_to_branches
        self.tree_objects = []
        self.query_made = False

        for name, objects in self.map_names_to_branches.items():
            if to_index in name:
                self.tree_objects.append(objects)


        self.tree_names = np.array([obj.name for obj in self.tree_objects], 
                                   dtype=StringDType)
        
        self.coords = np.array([obj.start_loc for obj in self.tree_objects], 
                               dtype=np.float64)

        if self.tree_names.size == 0:
            raise ValueError('Locations must be given')

        #create the data structure  
        self.tree = KDTree(self.coords)
    
    def query_objects_within_r(self, 
                               referance_name = 60, # Can be a name or index 
                               radius = 0.1,
                               output_type="dict"):
        self.query_made = True
        start_time = tm.perf_counter()
        if type(referance_name) is int:
            reference_index = referance_name
        else:
            reference_index = int(np.where(self.tree_names == referance_name)[0][0])
        
        self.query_points = self.coords[reference_index]
        self.neighbor_indices = np.asarray(self.tree.query_ball_point(self.query_points, 
                                                                      radius, 
                                                                      return_length=False), 
                                      dtype=np.int32)

        self.neighbor_points = self.coords[self.neighbor_indices]
        distances = np.linalg.norm(self.neighbor_points - self.query_points, axis=1)

        s_delta = tm.perf_counter() - start_time

        print(f"{self.neighbor_indices.size - 1} Neighbors found using data structure in {s_delta} sec")

        if output_type == 'dict':
            output = {}
            for name, location, distance in zip(self.tree_names[self.neighbor_indices], self.neighbor_points, distances):
                output[name] = {'location':location.tolist(), 'distance':float(distance)}
                output = DotDict(output)
        else:     
            # If you want the output to be a numpy array instead
            dtype = [('name', StringDType), 
                    ('location', np.float64, (3,)), 
                    ('distance', np.float64)]

            output = np.empty(self.tree_names[self.neighbor_indices].size, dtype=dtype)
            output['name'] = self.tree_names[self.neighbor_indices]
            output['location'] = self.neighbor_points
            output['distance'] = distances
        
        return output
    
    def find_edges(self):
        line_query = np.tile(self.query_points.tolist(), 
                                (self.neighbor_points[:, 0].size, 1))
        
        x_line = np.concatenate((line_query[:, [0]], self.neighbor_points[:, [0]]), axis=1)
        y_line = np.concatenate((line_query[:, [1]], self.neighbor_points[:, [1]]), axis=1)
        z_line = np.concatenate((line_query[:, [2]], self.neighbor_points[:, [2]]), axis=1)

        return x_line, y_line, z_line
    
    def make_plot(self, save_to= "data/nn_plot.png"):
        if self.query_made == False:
            raise ValueError("Query must be run to plot the referance and objects found")
        # Set up the 3D canvas
        plt.ion
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

            Line2D([0], [0], 
                marker='.', 
                color='w',  
                markerfacecolor='blue', 
                markersize=15, 
                label='w/ Iteration'),

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

    def test_time():
        return
        start_time = tm.perf_counter()
        for bud in all_buds:
            dist = get_dist(ref.start_loc, bud.start_loc)
            if dist < r:
                within_iter.append(bud.start_loc)
        end_time = tm.perf_counter()
        i_delta = end_time - start_time
        print(f"{len(within_iter)} Neighbors found using iteration in {i_delta} sec")
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

        t_ratio = height_m / trunk.growth.length
        
        diameter_at_height_m = trunk.growth.get_diameter(t=t_ratio)
        
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
                
            t_ratio = measurement_dist_m / length_m
            
            diameter_m = obj.growth.get_diameter(t=t_ratio)
            
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