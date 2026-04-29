import openalea.plantgl as plantgl
import openalea.plantgl.scenegraph as sg
import openalea.plantgl.algo as alg
from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention


# Convert the PlantGL to a list of vertices and faces
def plant_gl_scene_to_vertices_and_faces(scene):
    """ extract vertices and faces from a plantGL scene graph.
    helper function for creating ply and usd files"""
    d = alg.Discretizer()

    vertices = []  # List of point List
    colors = []  # List of colors
    texture_coords = []
    faces = []  # list  of tuple (offset,index List)

    counter = 0
    ret_dict = []
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
                flatten_f = list(map(lambda x: x + counter, j))
                mesh_component["faces"].append(flatten_f)
        counter += n
        ret_dict.append(mesh_component)
        if n != 16:
            print(f"Diff number of vs {n}")
    return ret_dict


def stitch_cylinders(mesh_components:dict, meta_data: dict)->dict:
    name_tree_mapping = meta_data["tree_mapping"]
    tree = meta_data["tree"]
    color_name_mapping = meta_data["color_mapping"]

    collect_components = {}
    collect_spur_components = {}
    for key, item in tree.part_list[TreeNamingConvention.part_names[TreeNamingConvention.TRUNK]].items():
        collect_components[key] = {"part_dict": item, "mesh_cyl": [], "spur": []}
    for key, item in tree.part_list[TreeNamingConvention.part_names[TreeNamingConvention.BRANCH]].items():
        collect_components[key] = {"part_dict": item, "mesh_cyl": [], "spur": []}
    for key, item in tree.part_list[TreeNamingConvention.part_names[TreeNamingConvention.SPUR]].items():
        collect_spur_components[key] = item

    for key, item in mesh_components.items():
        hierarchy_name = color_name_mapping[key]
        tree_part_name = name_tree_mapping[hierarchy_name]

        spur = False
        if tree_part_name in collect_spur_components:
            spur_dict = collect_spur_components[tree_part_name]
            spur = True
            get_c = collect_components[spur_dict["paret_name"]]
        else:
            get_c = collect_components[tree_part_name]
        if spur:
            get_c["spur"].append(item)
        else:
            get_c["mesh_cyl"].append(item)

    return collect_components


# PlantGL -> PLY
def write(fname, mesh_components: dict):
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
    for mesh_component in mesh_components:
        vertices.extend(mesh_component["vertices"])
        colors.extend(mesh_component["colors"])
        faces.extend(mesh_component["faces"])
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
end_header""".format(
        len(vertices), len(faces)
    )
    f.write(header + "\n")
    for pt, color in zip(vertices, colors):
        r, g, b = color
        x, y, z = pt
        f.write("{:.4f} {:.4f} {:.4f} {:.0f} {:.0f} {:.0f}\n".format(x, y, z, r, g, b))
    for face in faces:
        f.write("{:.0f}".format(len(face)))
        for a in face:
            f.write(" {}".format(a))
        f.write("\n")

    f.close()
    return


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
