import omni.replicator.core as rep
import omni.usd
import carb.settings
from pxr import UsdLux, Usd, UsdGeom, Vt, Gf, UsdSemantics, Sdf, UsdShade, Ar
from numpy import random


b_uv_render = True

if b_uv_render:
    carb_settings = carb.settings.get_settings()
    # All of this was suppose to help, but... it did nothing
    #carb_settings.set("/rtx/rendermode", "RayTracedLighting")
    #carb_settings.set("/rtx/minimal/mode", 1)
    
    #omni.kit.commands.execute("ChangeViewportRenderModeCommand",
    #                          render_mode="Texture Diffuse",
    #                          viewport_name="Viewport") 
    carb_settings.set("/rtx/sceneDb/ambientLightColor", (1, 1, 1))
    carb_settings.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
    carb_settings.set("rtx/directLighting/enabled", False)
    carb_settings.set("rtx/indirectLighting/enabled", False)
    carb_settings.set("rtx/shadows/enabled", False)
    carb_settings.set("rtx/post/ambientocclusion/enabled", False)
    carb_settings.set("rtx/pathtracing/cachedShadows/enabled", False)
    # Don't pause to collect frames
    carb.settings.get_settings().set("/omni/replicator/RTSubFrames", 1)
    # No antialiasing
    carb.settings.get_settings().set("/rtx/post/aa/op", 0)
    # Disable motion blur (pre and post)
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", False)
    carb.settings.get_settings().set("/rtx/post/motionBlur/scale", 0.0)
    # and reflections
    carb.settings.get_settings().set("/rtx/reflections/maxBounces", 0)
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
        
else:
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
        
# 1. Define the scene and assets
# Note: replace the USD paths below with your own 3D assets
tree_path = "/models/lpy_envy_00000.usda"
camera_path = "/World/Camera"
output_directory = "/home/cindy/VSCode/data/isaac_sim_output"

stage_dir = "/home/cindy/isaacsim/World"

omni.client.add_default_search_path(stage_dir)

def RightAngleCameras():
    camera_delta = 0.1
    camera_pos = [(0, -4, 2),(0, -4, 2.5),(0, -4, 3.0)]
    look_at_pos = []
    camera_left_pos = []
    camera_up_pos = []
    for pos in camera_pos:
        camera_left_pos.append((pos[0] - camera_delta, pos[1], pos[2]))
        camera_up_pos.append((pos[0], pos[1] + camera_delta, pos[2]))
        look_at_pos.append((0, 0, pos[2]))
    return camera_pos, camera_left_pos, camera_up_pos, look_at_pos

search_paths = [stage_dir]
stage = omni.usd.get_context().get_stage()

# 2. Create a context with these paths
stage_context = Ar.DefaultResolverContext(search_paths)

"""
# My nth attempt at reading in the material texture from the file so it will be found in tree. Pretty much failing
# read in the pinebark tetxure
# Spawn the object to be detected
material_prim_path = "/Materials/PineBark"
override_prim = stage.OverridePrim(material_prim_path)
with Ar.ResolverContextBinder(stage_context):
    override_prim.GetReferences().AddReference(assetPath="./textures/pine_bark.usda",
                                               primPath=material_prim_path)
    #pinebark = stage.create.from_usd("./textures/pine_bark.usda")
"""


with rep.new_layer():
    # Create a simple environment
    rep.create.plane(scale=10, visible=True)
    
    # Spawn the object to be detected
    with Ar.ResolverContextBinder(stage_context):
        #pinebark = rep.create.from_usd("./textures/pine_bark.usda")
        if b_uv_render:
            tree = rep.create.from_usd("./models/lpy_envy_00000_uv.usda", name="tree")
        else:
            tree = rep.create.from_usd("./models/lpy_envy_00000.usda", name="tree")

    #pinebark = rep.create.from_usd("./textures/pine_bark.usda")
    #all = rep.create.from_dir(stage_dir, True)
    
    # Spawn the camera and attach to a render product
    b_right_angle = False
    camera_pos, camera_left_pos, camera_up_pos, look_at_pos = RightAngleCameras()
    if b_right_angle:
        camera = rep.create.camera(position=camera_pos[0], look_at=look_at_pos[0])
        camera_left = rep.create.camera(position=camera_left_pos[0], look_at=look_at_pos[0])
        camera_up = rep.create.camera(position=camera_up_pos[0], look_at=look_at_pos[0])
        render_product = rep.create.render_product(camera, (512, 512), name="primary")
        render_product_left = rep.create.render_product(camera_left, (512, 512), name="left")
        render_product_up = rep.create.render_product(camera_up, (512, 512), name="up")
    else:
        camera = rep.create.camera(position=camera_pos[0], look_at=look_at_pos[0])
        render_product = rep.create.render_product(camera, (512, 512), name="solo")
        
    uv_color = omni.replicator.core.create.material_omnipbr(diffuse=(0.0, 0.0, 0.0),
                                                            roughness = 0.0,
                                                            diffuse_texture=stage_dir + "/textures/mesh_uv.png")

    if b_uv_render:
        #rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 0.0)
        #rep.settings.set_stage_lights(False)
        default_light_path = "/Environment/defaultLight"
        if stage.GetPrimAtPath(default_light_path):
            prim = stage.GetPrimAtPath(default_light_path)
            print(f"default light {prim.GetTypeName()}")
            geom_schema = UsdGeom.Imageable(prim)
            #geom_schema.MakeInvisible()
            
        """ THis doesn't work because there are no lights so all_lights is an empty replicator
        all_lights = rep.get.prims(path_pattern="**",
                                   prim_types=["DistantLight", "RectLight", "SphereLight", "CylinderLight", "DiskLight"])

        # Turn off default light, if there is one
        if len(all_lights) > 0:
            with all_lights:
                rep.modify.visibility(False)
        """
        """
        light_pos = (camera_pos[0][0] * 100, camera_pos[0][1] * 100, camera_pos[0][2] * 100)
        print(f"{light_pos} and {look_at_pos[0]}")
        light = rep.create.light(position=light_pos,
                                 look_at = look_at_pos[0],
                                 color=(1.0, 1.0, 1.0),
                                 #exposure=10.0,
                                 texture=None,
                                 name="DistantLight",
                                 intensity=1000, 
                                 light_type="DistantLight")
        """
        
        # 2. Create the ambient light source
        # A Dome Light provides uniform 360-degree environment lighting
        #black_light = rep.create.light(
        #    light_type="distance",
        #    intensity=0 # minimal mode uses at least one light; this turnis it off
        #    )    
        
    else:
        # Add Default Light
        light = rep.create.light(rotation=(0,0,0), intensity=3000, light_type="distant")

    #UsdShade.MaterialBindingAPI(tree).Bind(uv_color)

    #rep.settings.set_render_rtx_realtime(antialiasing="Off")

    # Get all the materials on all the primitives
    #target_materials = rep.get.material(path_pattern="*Looks*")
    if b_uv_render:
        with rep.get.prims():
            rep.modify.material(uv_color)
    
    # 2. Define domain randomization
    texture_filename_list = [stage_dir + "/textures/mesh_uv.png"]
    with rep.trigger.on_frame(max_execs=10):
        # Randomize the cube's position and rotation
        with tree:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.1, -0.2, -0.25), (0.1, 0.2, 0.25)),
                rotation=rep.distribution.uniform((-5, -15, 0), (5, 15, 0))
            )

        # Randomize the camera placement
        cam_seed = random.randint(10, 10000)
        if b_right_angle:
            with camera:
                rep.modify.pose(
                    #position=rep.distribution.uniform((-1, -5, 1), (1, 5, 3)),
                    position=rep.distribution.choice(camera_pos, seed=cam_seed, name="bottom_mid_top"),
                    look_at=rep.distribution.choice(look_at_pos, seed=cam_seed, name="bottom_mid_top")
                )
            with camera_left:
                rep.modify.pose(
                    #position=rep.distribution.uniform((-1, -5, 1), (1, 5, 3)),
                    position=rep.distribution.choice(camera_left_pos, seed=cam_seed, name="bottom_mid_top"),
                    look_at=rep.distribution.choice(look_at_pos, seed=cam_seed, name="bottom_mid_top")
                )
            with camera_up:
                rep.modify.pose(
                    #position=rep.distribution.uniform((-1, -5, 1), (1, 5, 3)),
                    position=rep.distribution.choice(camera_up_pos, seed=cam_seed, name="bottom_mid_top"),
                    look_at=rep.distribution.choice(look_at_pos, seed=cam_seed, name="bottom_mid_top")
                )
        else:
            with camera:
                pick_one = random.randint(0, len(camera_pos))
                cam_loc = camera_pos[pick_one]
                rep.modify.pose(
                    position=rep.distribution.uniform((cam_loc[0] - 1, cam_loc[1] - 5, cam_loc[2] - 1), 
                                                      (cam_loc[0] + 1, cam_loc[1] + 5, cam_loc[2] + 1)),
                    look_at=rep.distribution.choice(look_at_pos, seed=cam_seed, name="bottom_mid_top")
                )

    # 3. Set up the writer (saves to disk)
    # This writer saves RGB, Bounding Boxes, and Depth data
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        distance_to_camera=True,
        bounding_box_2d_loose=True
    )
    
    # Attach the writer to the render product and run
    if b_right_angle:
        writer.attach([render_product, render_product_left, render_product_up])
    else:
        writer.attach([render_product])
    
    # Execute the simulation
    rep.orchestration.run()
