import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


def parse_ply(ply_path):
    """
    Parses an ASCII PLY file and returns a dictionary mapping colors to a list of vertices.
    """
    vertices_by_color = defaultdict(list)

    with open(ply_path, "r") as f:
        lines = f.readlines()

    header_end_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end_index = i + 1
            break

    # Assuming vertex properties are x, y, z, diffuse_red, diffuse_green, diffuse_blue
    # based on helpers.py write function

    # We need to know how many vertices there are to stop reading before faces
    # But simpler is to just try to parse lines as vertices until we hit something else or end of vertices
    # The header says "element vertex N". Let's find N.

    num_vertices = 0
    for line in lines[:header_end_index]:
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[2])

    for i in range(header_end_index, header_end_index + num_vertices):
        parts = lines[i].split()
        if len(parts) < 6:
            continue

        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))

        color_key = f"({r}, {g}, {b})"
        vertices_by_color[color_key].append([x, y, z])

    return vertices_by_color


def sort_by_color(vs, cs):
    """
    Group the vertices by color
    """
    vertices_by_color = defaultdict(list)

    for v, c in zip(vs, cs):
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        r, g, b = c[0], c[1], c[2]

        color_key = f"({r}, {g}, {b})"
        vertices_by_color[color_key].append([x, y, z])

    return vertices_by_color


def compute_cylinder_params(vertices):
    """
    Computes centroid, radius, and length of a cylinder given its vertices.
    """
    points = np.array(vertices)

    if len(points) < 2:
        return np.mean(points, axis=0).tolist(), 0.0, 0.0

    # 1. Centroid
    centroid = np.mean(points, axis=0)

    # 2. Principal Axis via PCA
    centered_points = points - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Handle case where points are collinear or coplanar (singular matrix)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    except np.linalg.LinAlgError:
        # Fallback for degenerate cases
        return centroid.tolist(), 0.0, 0.0

    # The principal axis corresponds to the largest eigenvalue
    # eigh returns eigenvalues in ascending order
    principal_axis = eigenvectors[:, -1]

    # 3. Length
    # Project points onto the principal axis
    projections = np.dot(centered_points, principal_axis)
    length = np.max(projections) - np.min(projections)

    # 4. Radius
    # Calculate distance of points from the axis
    # Vector from centroid to point is 'centered_points'
    # Projection vector along axis is 'projections[:, np.newaxis] * principal_axis'
    # Perpendicular vector is 'centered_points - projection_vector'
    projection_vectors = np.outer(projections, principal_axis)
    perpendicular_vectors = centered_points - projection_vectors
    distances = np.linalg.norm(perpendicular_vectors, axis=1)

    radius = np.mean(distances)

    # 5. Rotation of the cylinder
    # The principal axis is the direction of the cylinder
    orientation = principal_axis.tolist()

    return {"centroid":centroid.tolist(), "radius":float(radius), "length":float(length), "orientation":orientation}


def get_cylinder_params(mesh_component: dict, cylinder_metadata: dict) -> dict:
    vertices_by_color = sort_by_color(mesh_component["vertices"], mesh_component["colors"])
    cylinder_params = {}
    for color_key, vertices in vertices_by_color.items():
        centroid, radius, length, orientation = compute_cylinder_params(vertices)

        cylinder_params[color_key] = {
            "part_name": cylinder_metadata[color_key]["part_name"],
            "centroid": centroid,
            "radius": radius,
            "length": length,
            "orientation": orientation,
        }

    return cylinder_params


def get_all_cylinder_params(mesh_all_components : list, cylinder_metadata: dict) -> dict:
    cylinder_params = {}
    for mesh_component in mesh_all_components:
        # cylinder_param = compute_cylinder_params(mesh_component["vertices"])
        mesh_component_params = get_cylinder_params(mesh_component, cylinder_metadata)
        for key, item in mesh_component_params.items():
            cylinder_params[key] = item
    return cylinder_params


def add_cylinder_params_to_json(ply_path, json_path, output_json_path=None):
    print(f"Processing {ply_path} and {json_path}...")

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Parse PLY
    vertices_by_color = parse_ply(ply_path)

    updated_count = 0

    # Update JSON
    for color_key, vertices in vertices_by_color.items():
        if color_key in data:
            centroid, radius, length, orientation = compute_cylinder_params(vertices)

            current_value = data[color_key]

            # Check if already updated (list) or raw string
            part_name = current_value if isinstance(current_value, str) else current_value["part_name"]

            data[color_key] = {
                "part_name": part_name,
                "centroid": centroid,
                "radius": radius,
                "length": length,
                "orientation": orientation,
            }
            updated_count += 1

    if output_json_path is None:
        output_json_path = json_path

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated {updated_count} entries in {output_json_path}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich JSON with cylinder parameters from PLY.")
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY file")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
    parser.add_argument("--output", type=str, help="Path to output JSON file (optional, defaults to overwriting input)")

    args = parser.parse_args()

    add_cylinder_params_to_json(args.ply, args.json, args.output)
