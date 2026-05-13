import openalea.plantgl as plantgl
import openalea.plantgl.scenegraph as sg
import openalea.plantgl.algo as alg

from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention
from lpy_treesim.color_manager import ColorManager
import numpy as np
from trimesh import Trimesh
from trimesh.visual import TextureVisuals


def stitch_cylinder(cyls: list, col_plant_type: tuple, col_instance: tuple) -> dict:
    """Takes in a list of cylinders (should be in order along the trunk/branch) and makes a single mesh out of it
    Merges the vertices from the previous row with the next to make the mesh seamless
    Adds texture map coordinates
    Applies the color for semantic labeling (eg branch, spur, trunk) to the vertex colors
    Applies the color for instance labeling (eg branch 32) to the face colors
    @param cyls - the list of cylinders with vertices, faces
    @param col_plant_type - the semantic color of the plant, from TreeNamingConvention
    @param col_instance - the semantic color for the instance, from TreeNamingConvention
    @return the stitched together mesh as vertices and faces, texture coordinates, and vertex/face colors"""
    mesh_component = {"vertices": [], "vertex_colors": [], "faces": [], "textures": [], "face_colors": []}
    offset = 0
    v_delta = 1.0 if len(cyls) == 1 else 1.0 / (len(cyls) - 1.0)
    v_coord = 0.0
    t_along = 0.5 * v_delta

    for cyl in cyls:
        n_split = len(cyl["vertices"]) // 2
        s_div = 1.0 / (n_split - 1.0)
        for vi in range(0, n_split):
            mesh_component["vertices"].append(cyl["vertices"][2 * vi])
            mesh_component["vertex_colors"].append(col_plant_type)
            mesh_component["textures"].append((vi * s_div, v_coord))
        v_coord += v_delta

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

    cyl = cyls[-1]
    for vi in range(0, len(cyl["vertices"]) // 2):
        n_split = len(cyl["vertices"]) // 2
        s_div = 1.0 / (n_split - 1.0)
        mesh_component["vertices"].append(cyl["vertices"][2 * vi])
        mesh_component["vertex_colors"].append(col_plant_type)
        mesh_component["textures"].append((vi * s_div, v_coord))
    return mesh_component




def stitch_cylinders(tree:TreeNamingConvention) -> (dict, list):
    keys_to_remove = []

    color_to_part = {"semantic":{}, "instance":{}}
    for part_dict in tree.iterate_all_wood_parts():
        if len(part_dict["mesh_cyl"]) > 0:
            col_plant_type = TreeNamingConvention.semantic_color(part_dict["name"])
            col_instance = tree.instance_color(part_dict["name"])
            color_to_part["semantic"][col_plant_type] = part_dict["full_name"]
            color_to_part["instance"][col_plant_type] = part_dict["full_name"]
            part_dict["mesh"] = stitch_cylinder(part_dict["mesh_cyl"],
                                                col_plant_type=col_plant_type,
                                                col_instance=col_instance)
        else:
            if part_dict['type'] == TreeNamingConvention._root_key():
                continue
            keys_to_remove.append(part_dict["full_name"])
            print(f"Part {part_dict['name']} has no mesh, removing")

    for key in keys_to_remove:
        tree.remove_key(key)
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
            mesh_component["vertices"].append(pt)
        for j in face:
            flatten_f = list(map(lambda x: x, j))
            mesh_component["faces"].append(flatten_f)
        tree_part_dict["mesh_cyl"].append(mesh_component)
        if n != 16:
            print(f"Diff number of vs {n}")
    print(f"Found {len(ret_list)} Cylinders")
    return ret_list


def stitch_cylinders_orig(mesh_components:list, meta_data: dict)->(dict, ColorManager):
    name_tree_mapping = meta_data["tree_mapping"]
    tree = meta_data["tree"]
    color_name_mapping = meta_data["color_mapping"]

    collect_components = {}
    for key, item in tree.part_list[TreeNamingConvention.part_names[TreeNamingConvention.TRUNK]].items():
        collect_components[key] = {"part_dict": item, "mesh_cyl": [], "spur": []}
    for key, item in tree.part_list[TreeNamingConvention.part_names[TreeNamingConvention.BRANCH]].items():
        collect_components[key] = {"part_dict": item, "mesh_cyl": [], "spur": []}

    for mc in mesh_components:
        key = mc["unique_id"]
        hierarchy_name = color_name_mapping[key]["part_name"]
        tree_part_name = name_tree_mapping[hierarchy_name]

        if tree_part_name["type"] == 'spur':
            parent_name = tree_part_name["parent_name"]
            if not parent_name in collect_components:
                continue
            get_c = collect_components[parent_name]
            get_c["spur"].append(mc)
        else:
            part_name = tree_part_name["name"]
            get_c = collect_components[part_name]
            get_c["mesh_cyl"].append(mc)

    make_colors = ColorManager()
    for key, item in collect_components.items():
        if len(item["mesh_cyl"]) == 0:
            print(f"Warning, part {key} empty")
        else:
            if len(item["mesh_cyl"]) > 1:
                tube = stitch_cylinder(item["mesh_cyl"], key, make_colors)
                item["mesh_cyl"] = [tube]
            else:
                cyl = item["mesh_cyl"][0]
                cyl["face_colors"] = []
                name_base = key
                t_along = 0.5
                s_div = 1.0 / (len(cyl["vertices"]) - 1.0)
                for ind, f in cyl["faces"]:
                    name = f"{name_base}-{t_along:0.2}-{ind*s_div:0.2}"
                    col = make_colors.get_unique_color(name=name)
                    cyl["face_colors"] = col
    return collect_components, make_colors


# from https://pymeshlab.readthedocs.io/en/latest/tutorials/import_mesh_from_arrays.html
def create_mesh(tree: TreeNamingConvention)->Trimesh:
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

    texs = TextureVisuals(uv=vs_tex_np)
    mesh = Trimesh(vertices=vs_np, faces=fs_np, vertex_colors=vs_cols_np, face_colors=fs_cols_np, visual=texs, process=False)
    # mesh_tex = trimesh.visual.texture.Tex
    return mesh


def write_mesh(fname: str, tree: TreeNamingConvention):
    mesh = create_mesh(tree)
    mesh.export(fname)


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
