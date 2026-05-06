from pxr import Usd, UsdGeom, Vt, Gf, UsdSemantics, Sdf, UsdShade, Ar
from lpy_treesim.tree_generation.naming_convention import TreeNamingConvention
from lpy_treesim.color_manager import ColorManager
import ctypes


def create_labeled_asset(file_path):
    # 1. Create a new Stage
    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 2. Create your Mesh or Xform
    root_path = "/MyLabeledObject"
    mesh_prim = UsdGeom.Mesh.Define(stage, root_path)

    # 3. Apply the SemanticsAPI
    # The 'instanceName' (second arg) is usually "class" for Isaac Sim
    semantics_api = UsdSemantics.SemanticsAPI.Apply(mesh_prim.GetPrim(), "class")

    # 4. Set the attributes that Isaac Sim's annotators read
    # semanticType: Defines the category (usually 'class')
    # semanticData: Defines the specific label (e.g., 'engine_part')
    semantics_api.CreateSemanticTypeAttr().Set("class")
    semantics_api.CreateSemanticDataAttr().Set("engine_part")

    # Do twice to get two different labelings
    semantics_api.CreateSemanticTypeAttr().Set("class II")
    semantics_api.CreateSemanticDataAttr().Set("engine_part")

    # Save the stage
    stage.GetRootLayer().Save()
    print(f"Asset created at: {file_path}")


def create_mesh(stage, path, points, face_vertex_counts, face_vertex_indices):
    """
    Helper function to create a USD mesh.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def check_texture(stage_context):
    with Ar.ResolverContextBinder(stage_context):
        # 1. Create a new USD stage
        # Set the up axis and units
        stage = Usd.Stage.CreateInMemory()
        #stage = Usd.Stage.CreateNew("test_texture.usda")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # 2. Define the Mesh primitive
        mesh = UsdGeom.Mesh.Define(stage, '/tree_texture_check')

        # 2. Define Texture Coordinates (UVs)
        # We use 'st' as the name.
        # 'interpolation' determines how UVs map to the geometry.
        tex_coords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.varying
        )

        # 2. Create the Material
        material_path = Sdf.Path("/textures/pine_bark_vmbibe2g_2k")
        material = UsdShade.Material.Define(stage, material_path)

        # 3. Create the Shader (UsdPreviewSurface)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PBRShader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)

        # 4. Create the Texture Sampler (UsdUVTexture)
        reader = UsdShade.Shader.Define(stage, material_path.AppendChild("TexSampler"))
        reader.CreateIdAttr("UsdUVTexture")
        reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("/textures/Pine_Bark_vmbibe2g_2K_BaseColor.jpg")
        # Connect texture output to shader's diffuseColor input
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f))

        # 5. Create the Primvar Reader (To tell the texture to use 'st')
        st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("STReader"))
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        # Connect reader output to texture sampler's st input
        reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))

        # 6. Bind the Material to the Mesh
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
        # 6. Save the stage
        #print(stage.GetRootLayer().ExportToString())
        stage.GetRootLayer().Export("tree_texture_check.usda")


def make_mesh_from_components(mesh_usd, mesh_parts):
    # Vertices, texture coords for vs, and faces
    vs = []
    vs_texs = []
    # Not really sure we need to do this, but otherwise have trouble with the USD call
    for pt, tex in zip(mesh_parts["vertices"], mesh_parts["textures"]):
        # swap y and z to make z up
        vs.append((pt[0], pt[2], pt[1]))
        vs_texs.append((tex[0], tex[1]))

    face_counts = []
    face_indices = []
    for face in mesh_parts["faces"]:
        face_counts.append(len(face))
        face_vs = []
        for idx in face:
            face_vs.append(idx)
        face_indices.append(face_vs)

    # I don't know if you need to do this, but it balks otherwise
    vs_ind_gen = [(pt[0], pt[2], pt[1]) for pt in vs]
    mesh_usd.CreatePointsAttr(vs_ind_gen)
    mesh_usd.CreateFaceVertexCountsAttr(face_counts)

    # Convert face_indices into a generator
    face_ind_gen = [idx for face in face_indices for idx in face]
    mesh_usd.CreateFaceVertexIndicesAttr(face_ind_gen)

    # Add Texture Coordinates (UVs)
    # We use 'st' as the name.
    # 'interpolation' determines how UVs map to the geometry.
    tex_coords = UsdGeom.PrimvarsAPI(mesh_usd).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.varying
    )
    # Set the UV values
    # These correspond to the points defined above: (u, v)
    # ts = [(t[0], t[1]) for t in mesh_component["textures"]]
    ts_ind_gen = [(pt[0], pt[1]) for pt in vs_texs]
    tex_coords.Set(ts_ind_gen)

    # No subdivision, please
    mesh_usd.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)


def setup_top_level_textures(stage):
    # 1. Create the Material at the top level
    material_path = Sdf.Path("/textures/pine_bark_vmbibe2g_2k")
    material = UsdShade.Material.Define(stage, material_path)

    # 2 Create the Shader (UsdPreviewSurface)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PBRShader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)

    # 3. Create the Texture Sampler (UsdUVTexture)
    reader = UsdShade.Shader.Define(stage, material_path.AppendChild("TexSampler"))
    reader.CreateIdAttr("UsdUVTexture")
    reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("../textures/Pine_Bark_vmbibe2g_2K_BaseColor.jpg")
    
    # 4 Connect texture output to shader's diffuseColor input
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f))

    # 5 Create the Primvar Reader (To tell the texture to use 'st')
    st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("STReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    
    # Connect reader output to texture sampler's st input
    reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))

    return material


def create_mesh_usd(stage_context, tree_name:str, path_tree_name:str, tree:TreeNamingConvention):
    # 1. Create a new USD stage
    # Set the up axis and units
    #stage = Usd.Stage.CreateNew("/World")
    # 3. Create or open your stage using this context
    with Ar.ResolverContextBinder(stage_context):
        stage = Usd.Stage.CreateInMemory()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # This acts as the "container" for your model
    root_xform = UsdGeom.Xform.Define(stage, f'/{tree_name}')

    # This fixes the "Cannot reference... has no default prim" error
    stage.SetDefaultPrim(root_xform.GetPrim())

    # Set up texture maps
    material = setup_top_level_textures(stage)    
 
    # Loop through all of the (organized) mesh components, adding meshes for each
    for part_dict in tree.iterate_all_wood_parts():
        mesh = part_dict["mesh"]

        if mesh is None:
            continue

        if len(mesh["vertices"]) == 0:
            print(f"Skipping {part_dict["name"]}, no mesh parts")
            continue

        # 1 Define the Mesh primitive
        usd_name = part_dict["usd_name"]
        xform_name = f"/{tree_name}{usd_name}"
        mesh_name = f"/{tree_name}{usd_name}/meshGeom"

        # Xform for the tree part
        branch_xform = UsdGeom.Xform.Define(stage, xform_name)
        # Mesh for the tree part
        mesh = UsdGeom.Mesh.Define(stage, mesh_name)
        #  Bind the Material to the Mesh
        UsdShade.MaterialBindingAPI(branch_xform).Bind(material)

        # Add semantic label
        labels_api = UsdSemantics.LabelsAPI.Apply(mesh.GetPrim(), "class")
        print(dir(labels_api))
        
        # Set the mesh's color based on what part it is
        # 'constant' means one value is used for the entire primitive
        color_primvar = mesh.CreateDisplayColorPrimvar(interpolation=UsdGeom.Tokens.constant)
        col_semantic = tree.semantic_color(part_dict["name"])
        col_vec = Gf.Vec3f(col_semantic[0] / 255.0, col_semantic[1] / 255.0, col_semantic[2] / 255.0)
        color_primvar.Set([col_vec])
        labels_api.CreateLabelsAttr().Set([part_dict["type"]])

        # Actually adds the vertices, faces, and texture map coords
        make_mesh_from_components(mesh, part_dict["mesh"])

    #  Save the stage
    print(f"Saving file to {str(path_tree_name)}")
    #print(stage.GetRootLayer().ExportToString())
    stage.GetRootLayer().Export(str(path_tree_name))

# Example Data
# verts = [(0,0,0), (1,0,0), (1,1,0), (0,1,0)] # 4 vertices
# faces = [(0,1,2), (0,2,3)]                  # 2 triangles forming a square

# create_mesh_usd("mesh_example.usda", verts, faces)
