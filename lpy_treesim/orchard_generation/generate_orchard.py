import os

from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate

FILE_NAME = "lpy_tree_pole_wire.usda"

lpy_assets = ["./models/ufo_tree.usda"]
wire_asset = ["../components/cylinder1m.usdc"]


def add_component(name, path, parent_prim, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1)):
    # Create a wrapper prim for transforms
    wrapper = UsdGeom.Xform.Define(stage, parent_prim.GetPath().AppendChild(name))

    # Add the referenced asset inside the wrapper
    asset_prim = stage.DefinePrim(wrapper.GetPath().AppendChild("Asset"))
    asset_prim.GetReferences().AddReference(path)

    # Xformable API for wrapper
    xform = UsdGeom.Xformable(wrapper)

    # ---- Translate ----
    translate_ops = xform.GetOrderedXformOps()
    translate_op = next((op for op in translate_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translate))

    # ---- Rotate (ZYX) ----
    rotate_op = next((op for op in translate_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateZYX), None)
    if rotate_op is None:
        rotate_op = xform.AddRotateZYXOp()
    rotate_op.Set(Gf.Vec3f(rotate[0], rotate[1], rotate[2]))

    # ---- Scale ----
    scale_op = next((op for op in translate_ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
    if scale_op is None:
        scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(*scale))

    return wrapper


def generate_orchard_rows_new(
        row_length,
        row_spacing,
        tree_spacing,
        row_numbers
):
    """
    Generate orchard trees as rows (list of lists).
    """
    orchard_rows = []
    # step 1: make one row of trees
    xs = np.arange(0, row_length, tree_spacing)

    # step 2: make an array of tree rows and assign coordinates
    ys = np.arange(0, row_numbers * row_spacing,
                   row_spacing)  # creates an array of equally spaced values within a defined interval

    # for y in range(0,row_numbers):
    for y in ys:
        row_points = []
        for x in xs:
            pt = Point(x, y)
            row_points.append(pt)
        orchard_rows.append(row_points)

    return (orchard_rows)


def generate_orchard_poles(
        row_length,
        row_spacing,
        pole_spacing,
        row_numbers
):
    orchard_poles = []
    # step 1: make one row of poles
    xs = np.arange(0, row_length, pole_spacing)

    # step 2: make an array of tree rows and assign coordinates
    ys = np.arange(0, row_numbers * row_spacing,
                   row_spacing)  # creates an array of equally spaced values within a defined interval

    for y in ys:
        pole_points = []
        for x in xs:
            pt = Point(x, y)
            pole_points.append(pt)
        orchard_poles.append(pole_points)

    return (orchard_poles)


def generate_orchard_wires(
        row_length,
        row_spacing,
        wire_height,
        row_numbers
):
    orchard_wires = []
    # step 1: make one row
    # xs = np.arange(0, row_length, wire_height[0])
    xs = np.arange(0, row_length, 1.0)  # 1.0 is the segment length x length of the usdc wire

    # step 2: make an array of tree rows and assign coordinates
    ys = np.arange(0, row_numbers * row_spacing,
                   row_spacing)  # creates an array of equally spaced values within a defined interval

    for y in ys:
        wire_points = []
        for x in xs:
            pt = Point(x, y)
            wire_points.append(pt)
        orchard_wires.append(wire_points)

    return (orchard_wires)


def generate_orchard_rows(
        origin,
        field_boundary,
        row_direction_deg,
        row_spacing,
        tree_spacing,
        clearance,
        min_trees_per_row=2
):
    """
    Generate orchard trees as rows (list of lists).
    """
    field_poly = Polygon(field_boundary)
    if not field_poly.is_valid:
        raise ValueError("Field boundary must form a valid polygon.")

    # orchard_poly = field_poly.buffer(-clearance)  #Negative distance buffers inside the polygon boundary (contracts it)
    orchard_poly = field_poly.buffer(0)  # Negative distance buffers inside the polygon boundary (contracts it)
    rotated_poly = rotate(orchard_poly, -row_direction_deg, origin=origin,
                          use_radians=False)  # rotates in negative direction
    # rotated_poly=rotated_poly.buffer(0.1)

    minx, miny, maxx, maxy = rotated_poly.bounds
    # print (rotated_poly.bounds)
    ys = np.arange(miny, maxy, row_spacing)  # creates an array of equally spaced values within a defined interval

    orchard_rows = []

    # row_num = 1

    for y in ys:
        row_points = []
        xs = np.arange(minx, maxx, tree_spacing)
        print(f"y= {y} , xs= {xs}")
        for x in xs:  # x is in the rotated along the individual trees
            pt = Point(x, y)
            # make polygon a bit bigger to include all points on the border
            # rotated_poly_buf = rotated_poly.buffer(0.1)
            # if rotated_poly.contains(pt):
            row_points.append(pt)
            if len(row_points) >= min_trees_per_row:
                # Rotate row back to original orientation
                row_points_rotated = [rotate(p, row_direction_deg, origin=origin, use_radians=False) for p in
                                      row_points]
                orchard_rows.append(row_points_rotated)

    return orchard_rows, orchard_poly


def generate_hedge(field_boundary, spacing, gaps=None):
    poly = Polygon(field_boundary)
    line = LineString(poly.exterior.coords)
    total_length = line.length
    num_trees = int(total_length // spacing)

    hedge_points = []
    for i in range(num_trees + 1):
        d = i * spacing
        frac = d / total_length
        if gaps:
            in_gap = any(g0 <= frac <= g1 for g0, g1 in gaps)
            if in_gap:
                continue
        pt = line.interpolate(d)
        hedge_points.append(pt)
    return hedge_points

"""
import omni.replicator.core as rep

# Create the render product
rp = rep.create.render_product("/OmniverseKit_Persp", (1024, 1024))

# Option 1: Render based on the 'category' scheme
segmentation = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
segmentation.initialize(semanticTypes=["category"])

# Option 2: To switch, you would change the semanticTypes to 'part_id'
# segmentation.initialize(semanticTypes=["part_id"])
"""

if __name__ == "__main__":
    os.chdir("/home/cindy/isaacsim/World/")
    origin = (1, 1)

    # Orchard tree parameters
    row_length = 30.0
    row_spacing = 3.0
    tree_spacing = 1.5  # 1.5
    row_numbers = 10

    pole_spacing = 4.0

    wire_height = [0.5, 0.9, 1.3, 1.7]

    n_trees = 10

    clearance = 0.0
    min_trees_per_row = 3
    row_direction_deg = 0.0

    # field boundary in meters
    field_boundary = field_boundary = [
        # (0, 0), (69, 0), (69, 38), (0, 38), (0, 0)
        # (0, 0), (6, 0), (6, 30), (0, 30)
        origin, ((row_numbers * row_spacing), origin[1]), ((row_numbers * row_spacing), row_length),
        (origin[0], row_length)
    ]

    # Hedge tree parameters
    hedge_spacing = 1.5
    # hedge_gaps = [(0.85, 0.86)]
    hedge_gaps = [(0.48, 0.83)]

    # Generate orchard rows
    # orchard_rows, orchard_poly = generate_orchard_rows(
    #     origin, field_boundary, row_direction_deg,
    #     row_spacing, tree_spacing, clearance, min_trees_per_row
    # )

    orchard_rows = generate_orchard_rows_new(
        row_length, row_spacing, tree_spacing, row_numbers
    )
    # Flatten for total count if needed
    all_trees = [tree for row in orchard_rows for tree in row]

    orchard_poles = generate_orchard_poles(
        row_length, row_spacing, pole_spacing, row_numbers
    )
    all_poles = [pole for row in orchard_poles for pole in row]

    orchard_wires = generate_orchard_wires(
        row_length, row_spacing, wire_height, row_numbers
    )
    all_wires = [wire for row in orchard_wires for wire in row]

    # Generate hedge
    # hedge_trees = generate_hedge(field_boundary, hedge_spacing, gaps=hedge_gaps)
    hedge_trees = []

    # Plot and save
    plt.figure(figsize=(10, 8))
    # x, y = orchard_poly.exterior.xy
    # plt.plot(x, y, 'k-', label="Field boundary (with clearance)")

    # Orchard trees
    for row in orchard_rows:
        plt.scatter([p.x for p in row], [p.y for p in row], s=20, c='green')
        # Optional: mark first and last tree of each row
        if row:
            plt.scatter([row[0].x], [row[0].y], s=40, c='blue', marker='o',
                        label="First tree" if row == orchard_rows[0] else "")
            plt.scatter([row[-1].x], [row[-1].y], s=40, c='red', marker='x',
                        label="Last tree" if row == orchard_rows[0] else "")

    # Poles
    for row in orchard_poles:
        plt.scatter([p.x for p in row], [p.y + 0.2 for p in row], s=10, c='brown')

    # plt.scatter([p.x for p in orchard_poles], [p.y for p in orchard_poles], s=40, c='brown', marker="s", label="Poles")

    # Hedge trees
    # plt.scatter([p.x for p in hedge_trees], [p.y for p in hedge_trees], s=40, c='brown', marker="s", label="Hedge trees")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Apple Orchard")
    plt.savefig("orchard_rows_wire.png", dpi=300)
    print(f"Saved png with {len(all_trees)} orchard trees, {len(hedge_trees)} hedge trees")

    # Create the USD stage
    stage = Usd.Stage.CreateNew(FILE_NAME)
    # UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Orchard")

    # row1 = UsdGeom.Xform.Define(stage, root)

    # Set this prim as the default prim
    stage.SetDefaultPrim(root.GetPrim())

    # for i, p in enumerate(hedge_trees):
    #    add_component(f"Hedge_{i:03}", hedge_assets[0], root, translate=(p.x, p.y, 0), rotate=(0, 0, np.random.uniform(0, 360)), scale=(np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)))
    rng_tree = np.random.default_rng()
    for i, p in enumerate(all_trees):
        tree_id = rng_tree.integers(0, 1, size=1)
        tree_str = f"{tree_id[0]:05}"
        other_side = 0.0 if np.random.uniform(0.0, 1.0, 1) else -180.0
        flip_lr = -1.0 if np.random.uniform(0.0, 1.0, 1) else -1.0
        add_component(f"Tree_{i:03}", f"./models/lpy_envy_{tree_str}.usda", root, translate=(p.x, p.y, 0),
                      rotate=(90, 0, other_side + np.random.uniform(-5, 5)),
                      scale=(flip_lr * np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)))

    for i, p in enumerate(all_poles):
        add_component(f"Pole_{i:03}", "./models/pole_wood.usdc", root, translate=(p.x, p.y, 0),
                      rotate=(np.random.uniform(0, 10), np.random.uniform(0, 10), 0),
                      scale=(np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)))

    for i, p in enumerate(all_wires):
        add_component(f"Wire_1_{i:03}", "./models/wire1m.usdc", root, translate=(p.x, p.y, wire_height[0]),
                      rotate=(0, 0, 0), scale=(1, 1, 1))
        add_component(f"Wire_2_{i:03}", "./models/wire1m.usdc", root, translate=(p.x, p.y, wire_height[1]),
                      rotate=(0, 0, 0), scale=(1, 1, 1))
        add_component(f"Wire_3_{i:03}", "./models/wire1m.usdc", root, translate=(p.x, p.y, wire_height[2]),
                      rotate=(0, 0, 0), scale=(1, 1, 1))
        add_component(f"Wire_4_{i:03}", "./models/wire1m.usdc", root, translate=(p.x, p.y, wire_height[3]),
                      rotate=(0, 0, 0), scale=(1, 1, 1))

    for i, p in enumerate(all_wires):
        add_component(f"Irrigation_{i:03}", "./models/irrigigation_pipe_bent.usdc", root,
                      translate=(p.x, p.y, wire_height[0] - 0.02), rotate=(np.random.uniform(0, 15), 0, 0),
                      scale=(1, 1, 1))

    stage.GetRootLayer().Save()
    print("USD created!")
