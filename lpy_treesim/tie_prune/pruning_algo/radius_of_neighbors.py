import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
import numpy as np
from math import sqrt
import time as tm
from pathlib import Path
import pandas as pd

def get_dist(L1, L2):
    return sqrt((L1.x-L2.x)**2+(L1.y-L2.y)**2+(L1.z-L2.z)**2)

def run_three_d(tree_sim_base, 
                branch_hierarchy, 
                map_names_to_branches, 
                k_neighbors=2, 
                distance_threshold=0.1):
    
    all_buds = []
    for name, branch in map_names_to_branches.items():
        if "bud" in name:
            all_buds.append(branch)
    if len(all_buds) == 0:
        print('[ERROR] NO BUD OBJECTS')

    # Unpack our custom location objects into a raw numpy array. 
    # SciPy is strict and only accepts pure numbers.
    coords = np.array([[loc.start_loc.x, loc.start_loc.y, loc.start_loc.z] for loc in all_buds])
    
    # Cap 'k' just in case we have fewer buds than the requested neighbor count
    k = min(k_neighbors, len(coords))
    
    # We need at least two points to draw a line between them
    if k < 2:
        print("Not enough marked locations to perform KNN.")
        return

    # Build the 3D spatial map of all the dots
    tree = KDTree(coords)
    
    query_num = 5
    r = 5
    within_iter = []
    within_n = []
    ref = all_buds.pop(query_num)

    start_time = tm.perf_counter()
    for bud in all_buds:
        dist = get_dist(ref.start_loc, bud.start_loc)
        if dist < r:
            within_iter.append(bud.start_loc)
    end_time = tm.perf_counter()
    i_delta = end_time - start_time
    print(f"{len(within_iter)} Neighbors found using iteration in {i_delta} sec")

    start_time = tm.perf_counter()

    #Find NN for one bud
    within_n = tree.query_ball_point(ref.start_loc, r, return_length=False)

    end_time = tm.perf_counter()
    s_delta = end_time - start_time
    print(f"{len(within_n)-1} Neighbors found using data structure in {s_delta} sec")
    print(f"Data structure is {round(((i_delta-s_delta)/i_delta)*100, 4)}% faster")
    
    # Ask the map to find the distances and ID numbers of the closest neighbors for every single dot
    #distances, indices = tree.query(within_iter, k=k)

    # Set up the 3D canvas
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    '''
    # Drop all the buds onto the plot as blue dots first
    ax.scatter(within_iter[:, 0], within_iter[:, 1], within_iter[:, 2], c='blue', s=25, label='Marked Vectors')

    # Go through every single dot to draw the connection lines
    for i in range(len(within_iter)):
        # Start at rank 1 because rank 0 is just the distance to itself (which is always 0)
        for rank in range(1, k):
            neighbor_idx = indices[i][rank]
            dist = distances[i][rank]
            # Grab the X, Y, Z pairs to draw a line between the current dot and this specific neighbor
            x_line = [coords[i, 0], coords[neighbor_idx, 0]]
            y_line = [coords[i, 1], coords[neighbor_idx, 1]]
            z_line = [coords[i, 2], coords[neighbor_idx, 2]]

            # If the neighbor is under our threshold, flag the connection in red.
            # Otherwise, color it green so we know it's a safe distance.
            if dist < distance_threshold:
                ax.plot(x_line, y_line, z_line, c='red', alpha=0.6)
            else:
                ax.plot(x_line, y_line, z_line, c='green', alpha=0.3)
    '''

    
    for l in within_iter:
        ax.scatter(l.x, l.y, l.z, 
                   c='blue', s=80, 
                   marker = ".")
    
    for index in within_n:
        ax.scatter(coords[index][0], coords[index][1], coords[index][2], 
                   c='green', s=12,
                   marker = "x")
    
        x_line = [ref.start_loc.x, coords[index][0]]
        y_line = [ref.start_loc.y, coords[index][1]]
        z_line = [ref.start_loc.z, coords[index][2]]
        ax.plot(x_line, y_line, z_line, c='red', alpha=0.2, lw = 0.6)

    ax.scatter(ref.start_loc.x, ref.start_loc.z, ref.start_loc.y, c='green', s=85, marker='.')

    ax.set_title(f"NN Evaluation\nData structure is {round(((i_delta-s_delta)/i_delta)*100, 4)}% faster")
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

    # Added this code in order to show what's generated by the plot within the sim
    edge_data = []
    for index in within_n:
        edge_data.append({
            'x1': ref.start_loc.x, 'y1': ref.start_loc.y, 'z1': ref.start_loc.z,
            'x2': coords[index][0], 'y2': coords[index][1], 'z2': coords[index][2]
        })
    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv("data/knn_edges.csv", index=False)

    # Save the nodes (The Buds)
    node_data = [{'x': ref.start_loc.x, 'y': ref.start_loc.y, 'z': ref.start_loc.z, 'type': 'ref'}]
    for index in within_n:
        node_data.append({
            'x': coords[index][0], 'y': coords[index][1], 'z': coords[index][2], 'type': 'neighbor'
        })
    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv("data/knn_nodes.csv", index=False)
    # Added this, end of new code changes 
    
    # Force an export of the image directly to the folder.
    # This acts as a backup in case the environment blocks the pop-up window.
    
    plt.savefig("data/nn_plot.png", dpi=300, bbox_inches='tight')
    print("Saved 3D plot to knn_evaluation.png")
    
    #plt.show()