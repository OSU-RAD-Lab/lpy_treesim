import openalea.plantgl as plantgl
import openalea.plantgl.scenegraph as sg
import openalea.plantgl.algo as alg
from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention
from lpy_treesim.color_manager import ColorManager


# Convert the PlantGL to a list of vertices and faces
def plant_gl_scene_to_vertices_and_faces(scene) ->list:
    """ extract vertices and faces from a plantGL scene graph.
    helper function for creating ply and usd files"""
    d = alg.Discretizer()

    vertices = []  # List of point List
    colors = []  # List of colors
    texture_coords = []
    faces = []  # list  of tuple (offset,index List)

    ret_list = []
    for item in scene:
        if not item.apply(d):
            continue

        p = d.result
        if isinstance(p, plantgl.scenegraph._pglsg.PointSet):
            continue

        mesh_component = {"vertices":[], "colors":[], "faces":[], "textures":[]}
        pts = p.pointList
        face = p.indexList
        n = len(p.pointList)
        n_around = n / 2
        if n > 0:
            color = item.appearance.diffuseColor()
            r, g, b = color
            mesh_component["unique_id"] = f"({r}, {g}, {b})"
            for v_id, pt in enumerate(pts):
                u = (v_id // 2) / (n_around - 1.0)
                v = (v_id % 2)
                mesh_component["vertices"].append(pt)
                mesh_component["colors"].append((r, g, b))
                mesh_component["textures"].append((u, v))
            for j in face:
                flatten_f = list(map(lambda x: x, j))
                mesh_component["faces"].append(flatten_f)
        ret_list.append(mesh_component)
        if n != 16:
            print(f"Diff number of vs {n}")
    print(f"Found {len(ret_list)} Cylinders")
    return ret_list


def stitch_cylinder(cyls: list, name_base: str, make_colors: ColorManager)->dict:
    mesh_component = {"vertices":[], "colors":[], "faces":[], "textures":[], "face_colors":[]}
    offset = 0    
    v_delta = 1.0 / (len(cyls) - 1.0)
    v_coord = 0.0
    t_along = 0.5 * v_delta
    for cyl in cyls:
        n_split = len(cyl["vertices"]) // 2
        s_div = 1.0 / (n_split - 1.0)
        for vi in range(0, n_split):
            mesh_component["vertices"].append(cyl["vertices"][2 * vi])
            mesh_component["colors"].append(cyl["colors"][2 * vi])
            mesh_component["textures"].append((vi * s_div, v_coord))
        v_coord += v_delta
        
        for ind, fi in enumerate(cyl["faces"]):
            face = []
            for id in fi:
                n_around = id // 2
                which_side = id % 2
                face.append(offset + which_side * n_split + n_around)

            mesh_component["faces"].append(face)
            name = f"{name_base}-{t_along:0.2}-{ind * s_div:0.2}"
            col = make_colors.get_unique_color(name=name)
            mesh_component["face_colors"].append(col)

        offset += len(cyl["vertices"]) // 2
        #print(f"v {cyl['vertices'][0]} {cyl['vertices'][1]}")
    
    cyl = cyls[-1]
    for vi in range(0, len(cyl["vertices"]) // 2):
        mesh_component["vertices"].append(cyl["vertices"][2 * vi])
        mesh_component["colors"].append(cyl["colors"][2 * vi])
        mesh_component["textures"].append((vi * s_div, v_coord))
    print(f"n vertices {len(mesh_component["vertices"])}, offset {offset}, max f {mesh_component["faces"][-8:]}")
    return mesh_component


def stitch_cylinders(mesh_components:list, meta_data: dict)->dict:
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
