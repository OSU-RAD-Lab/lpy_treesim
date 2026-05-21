import openalea.plantgl as plantgl
import openalea.plantgl.scenegraph as sg
import openalea.plantgl.algo as alg

from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention
from lpy_treesim.color_manager import ColorManager
from lpy_treesim.tree_generation.skeleton_convention import SkeletonComponent
import numpy as np
from trimesh import Trimesh
from trimesh.visual import TextureVisuals


def stitch_cylinder(skel: SkeletonComponent, cyls: list, col_plant_type: tuple, col_instance: tuple) -> dict:
    """Takes in a list of cylinders (should be in order along the trunk/branch) and makes a single mesh out of it
    Merges the vertices from the previous row with the next to make the mesh seamless
    Adds texture map coordinates
    Applies the color for semantic labeling (eg branch, spur, trunk) to the vertex colors
    Applies the color for instance labeling (eg branch 32) to the face colors
    @param skel - skeleton component to store the centers and radii in
    @param cyls - the list of cylinders with vertices, faces
    @param col_plant_type - the semantic color of the plant, from TreeNamingConvention
    @param col_instance - the semantic color for the instance, from TreeNamingConvention
    @return the stitched together mesh as vertices and faces, texture coordinates, and vertex/face colors"""

    # Vertices and faces come from the original cylinders; only keep the first ring of vertices from each
    #   cylinder to mesh is stitched together
    # Vertex colors are set by plont part type, face colors by a unique color for each instance
    # In the first draft, the u coordinates of the texture are set correctly but the v (vertical) are not
    mesh_component = {"vertices": [], "vertex_colors": [], "faces": [], "textures": [], "face_colors": [], "scale_texture":1.0}
    offset = 0
    v_delta = 1.0 if len(cyls) == 1 else 1.0 / (len(cyls) - 1.0)
    v_coord = 0.0
    n_split = 8   # Will be over-ridden

    for cyl in cyls:
        # mesh cylinders that come out of lpy alternate vertices in each ring
        n_split = len(cyl["vertices"]) // 2
        s_div = 1.0 / (n_split - 1.0)
        for vi in range(0, n_split):
            mesh_component["vertices"].append(cyl["vertices"][2 * vi])
            mesh_component["vertex_colors"].append(col_plant_type)
            mesh_component["textures"].append((vi * s_div, v_coord))
        v_coord += v_delta

        skel.add_cylinder(cyl["vertices"])

        # Re-do the face vertices so they do ring one ring two
        #   Note that these are quads. TriMesh will split to triangles
        for ind, fi in enumerate(cyl["faces"]):
            face = []
            for id in fi:
                n_around = id // 2
                which_side = id % 2
                face.append(offset + which_side * n_split + n_around)

            mesh_component["faces"].append(face)
            mesh_component["face_colors"].append(col_instance)

        offset += len(cyl["vertices"]) // 2
        # print(f"v {cyl['vertices'][0]} {cyl['vertices'][1]}")

    # And add in the last ring of vertices
    cyl = cyls[-1]
    for vi in range(0, len(cyl["vertices"]) // 2):
        n_split = len(cyl["vertices"]) // 2
        s_div = 1.0 / (n_split - 1.0)
        mesh_component["vertices"].append(cyl["vertices"][2 * vi])
        mesh_component["vertex_colors"].append(col_plant_type)
        mesh_component["textures"].append((vi * s_div, v_coord))

    # Now fix the t texture values so they are roughly spaced based on the length
    skel.compute_t_values()   # Calculate the length and t values based on each cylinder
    radii = 0.5 * (skel.radii[0] + skel.radii[-1])  # Average radius
    circum = 2.0 * np.pi * radii
    scl_t_values = skel.length / (2.0 * circum)   # Texture is twice as tall as wide
    tex_offset = np.random.uniform(0.0, 1.0)
    for n_rings, t_val in enumerate(skel.t_values):
        # This does a random offset of the texture
        # tex_v_value = tex_offset + t_val * scl_t_values
        # This sets the t values to be 0 to 1 along the cylinder
        tex_v_value = t_val 
        for indx in range(0, n_split):
            tex_coord = mesh_component["textures"][n_rings * n_split + indx]
            mesh_component["textures"][n_rings * n_split + indx] = (tex_coord[0], tex_v_value)

    return mesh_component


def stitch_cylinders(tree:TreeNamingConvention) -> (dict, list):
    keys_to_remove = []

    color_to_part = {"semantic":{}, "instance":{}}
    for part_dict in tree.iterate_all_wood_parts():
        if len(part_dict["mesh_cyl"]) > 0:
            col_plant_type = TreeNamingConvention.semantic_color(part_dict["name"])
            col_instance = tree.instance_color(part_dict["name"])
            color_to_part["semantic"][str(col_plant_type)] = part_dict["full_name"]
            color_to_part["instance"][str(col_instance)] = part_dict["full_name"]
            part_dict["mesh"] = stitch_cylinder(part_dict["skel"],
                                                part_dict["mesh_cyl"],
                                                col_plant_type=col_plant_type,
                                                col_instance=col_instance)
        else:
            if part_dict['type'] == TreeNamingConvention._root_key():
                continue
            keys_to_remove.append(part_dict["name"])
            print(f"Part {part_dict['name']} has no mesh, removing")

    return color_to_part, keys_to_remove


# Convert the PlantGL to a list of vertices and faces
def plant_gl_scene_to_vertices_and_faces(scene, tree: TreeNamingConvention, tree_mapping: dict, color_mapping:ColorManager) ->list:
    """ extract vertices and faces from a plantGL scene graph.
    helper function for creating ply and usd files"""
    d = alg.Discretizer()

    vertices = []  # List of point List
    faces = []  # list  of tuple (offset,index List)

    ret_list = []
    for item in scene:
        if not item.apply(d):
            continue

        p = d.result
        if isinstance(p, plantgl.scenegraph._pglsg.PointSet):
            continue

        pts = p.pointList
        face = p.indexList
        n = len(p.pointList)
        if n == 0:
            print(f"Empty cylinder")
            continue

        color = item.appearance.diffuseColor()
        # print(f"Name {name} id {id} color {color}")
        r, g, b = color
        unique_color = (r, g, b)
        hierarchy_name = color_mapping.color_to_name[unique_color]
        tree_part_dict = tree_mapping[hierarchy_name]
        mesh_component = {"vertices":[], "faces":[]}
        for v_id, pt in enumerate(pts):
            # pt_swap_y_z = [pt[0], pt[2], pt[1]]
            mesh_component["vertices"].append(pt)
        for j in face:
            flatten_f = list(map(lambda x: x, j))
            mesh_component["faces"].append(flatten_f)
        tree_part_dict["mesh_cyl"].append(mesh_component)
        if n != 16:
            print(f"Diff number of vs {n}")
    print(f"Found {len(ret_list)} Cylinders")
    return ret_list


# from https://pymeshlab.readthedocs.io/en/latest/tutorials/import_mesh_from_arrays.html
def create_mesh(tree: TreeNamingConvention, image_name)->(Trimesh, Trimesh, Trimesh):
    vs = []
    vs_tex = []
    vs_col = []
    faces = []
    face_cols = []
    v_offset = 0
    for part_dict in tree.iterate_all_wood_parts():
        mc = part_dict["mesh"]
        if mc == None:
            continue

        for v in mc["vertices"]:
            vs.append(v)
        for t in mc["textures"]:
            vs_tex.append(t)
        for c in mc["vertex_colors"]:
            # trimesh likes alpha
            vs_col.append([c[0], c[1], c[2], 255])
        for f, f_col in zip(mc["faces"], mc["face_colors"]):
            face = []
            for fid in f:
                face.append(fid + v_offset)
            face_cols.append([f_col[0], f_col[1], f_col[2], 255])
            faces.append(face)
            if len(face) == 4:
                # trimesh will split faces on load
                face_cols.append([f_col[0], f_col[1], f_col[2], 255])

        v_offset += len(mc["vertices"])

    vs_np = np.array(vs, dtype=np.float64)
    vs_tex_np = np.array(vs_tex, dtype=np.float32)
    vs_cols_np = np.array(vs_col, dtype=np.uint8)
    fs_np = np.array(faces, dtype=np.int64)
    fs_cols_np = np.array(face_cols, dtype=np.uint8)
    print(f"N vertices {vs_np.shape[0]}")
    print(f"N faces {fs_np.shape[0]} {fs_cols_np.shape}max {np.max(fs_np)}")

    texs = TextureVisuals(uv=vs_tex_np, image=image_name)
    mesh_uv = Trimesh(vertices=vs_np, faces=fs_np, visual=texs, process=False)
    mesh_fc = Trimesh(vertices=vs_np, faces=fs_np, face_colors=fs_cols_np, process=False)
    mesh_vc = Trimesh(vertices=vs_np, faces=fs_np, vertex_colors=vs_cols_np, process=False)
    # mesh_tex = trimesh.visual.texture.Tex
    return mesh_uv, mesh_fc, mesh_vc


def write_mesh(fname: str, tree: TreeNamingConvention, image_name: str):
    mesh_uv, mesh_fc, mesh_vc = create_mesh(tree=tree, image_name=image_name)
    mesh_uv.export(str(fname) + "_tm.obj")
    mesh_fc.export(str(fname) + "_fc.ply")
    mesh_fc.export(str(fname) + "_fc.obj")
    mesh_vc.export(str(fname) + "_vc.ply")


# PlantGL -> PLY
def write(fname, mesh_components: list):
    """Write a PLY file from a plantGL scene graph.
    This method will convert a PlantGL scene graph into an OBJ file.
    It does not manage  materials correctly yet.
    :Examples:
        import openalea.plantgl.scenegraph as sg
        scene = sg.Scene()"""

    # print("Write "+fname)
    f = open(fname, "w")

    vertices = []
    colors = []
    faces = []
    face_colors = []
    for mesh_component in mesh_components:
        vertices.extend(mesh_component["vertices"])
        colors.extend(mesh_component["colors"])
        faces.extend(mesh_component["faces"])
        face_colors.extend(mesh_component["face_colors"])
    header = """ply
format ascii 1.0
comment author abhinav
comment File Generated with PlantGL 3D Viewer
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {}
property list uchar int vertex_indices 
property uchar red
property uchar green
property uchar blue
end_header""".format(
        len(vertices), len(faces)
    )
    f.write(header + "\n")
    for pt, color in zip(vertices, colors):
        r, g, b = color
        x, y, z = pt
        f.write("{:.4f} {:.4f} {:.4f} {:.0f} {:.0f} {:.0f}\n".format(x, y, z, r, g, b))
    for face, face_col in zip(faces, face_colors):
        f.write("{:.0f}".format(len(face)))
        for a in face:
            f.write(" {}".format(a))
        r, g, b = color
        f.write("{:.0f} {:.0f} {:.0f}", r, g, b)
        f.write("\n")

    f.close()


def convert_ply_to_ext(in_path, out_path):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)
    ms.save_current_mesh(out_path)


# def convert_ply_to_x3d(in_path, out_path):
#     import pymeshlab
#     ms = pymeshlab.MeshSet()
#     ms.load_new_mesh(in_path)
#     ms.save_current_mesh(out_path)
