import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np

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
    
    # Ask the map to find the distances and ID numbers of the closest neighbors for every single dot
    distances, indices = tree.query(coords, k=k)

    # Set up the 3D canvas
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Drop all the buds onto the plot as blue dots first
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=25, label='Marked Vectors')

    # Go through every single dot to draw the connection lines
    for i in range(len(coords)):
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

    ax.set_title("KNN Distance Evaluation of Marked Locations")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Force an export of the image directly to the folder.
    # This acts as a backup in case the environment blocks the pop-up window.
    plt.savefig("knn_evaluation.png", dpi=300, bbox_inches='tight')
    print("Saved 3D plot to knn_evaluation.png")
    
    #plt.show()