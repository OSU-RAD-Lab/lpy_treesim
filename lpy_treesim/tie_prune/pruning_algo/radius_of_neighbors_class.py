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