import omni.replicator.core as rep
import omni.usd
import carb.settings
from pxr import Usd, UsdGeom, Vt, Gf, UsdSemantics, Sdf, UsdShade, Ar
from numpy import random


# Set DLSS execution mode to 2 (Quality) or 3 (Auto)
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

# 2. Create a context with these paths
stage_context = Ar.DefaultResolverContext(search_paths)
stage = Usd.Stage.CreateInMemory()
root_layer = stage.GetRootLayer()
root_layer.subLayerPaths.append("./models/lpy_envy_00000.usda")
root_layer.subLayerPaths.append("./texture/pine_bark.usda")
stage.Export("./models/compiled_scene.usda")

material_list = [
    "/Materials/PineBark",
]


with rep.new_layer():
    # Create a simple environment
    rep.create.plane(scale=10, visible=True)
    
    # Spawn the object to be detected
    tree = rep.create.from_usd("./models/lpy_envy_00000.usda", name="tree")

    #pinebark = rep.create.from_usd("./textures/pine_bark.usda")
    #all = rep.create.from_dir(stage_dir, True)
    
    b_flat_light = False
    if b_flat_light:
        #rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 0.0)
        rep.settings.carb_settings("/rtx/sceneDb/ambientLightColor", (0.0, 0.0, 0.0))
        #rep.settings.set_stage_lights(False)
        
        # 2. Create the ambient light source
        # A Dome Light provides uniform 360-degree environment lighting
        light = rep.create.light(
            light_type="Dome",
            intensity=1000,
            color=(1.0, 1.0, 1.0) # Pure white
            )    
    else:
        # Add Default Light
        light = rep.create.light(rotation=(0,0,0), intensity=3000, light_type="distant")

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
        
    uv_color = omni.replicator.core.create.material_omnipbr(diffuse=(0, 0, 0),
                                                            diffuse_texture=stage_dir + "/textures/mesh_uv.png")
    #UsdShade.MaterialBindingAPI(tree).Bind(uv_color)

    #rep.settings.set_render_rtx_realtime(antialiasing="Off")

    # 2. Define domain randomization
    with rep.trigger.on_frame(max_execs=10):
        # Randomize the cube's position and rotation
        with tree:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.1, -0.2, -0.25), (0.1, 0.2, 0.25)),
                rotation=rep.distribution.uniform((-5, -15, 0), (5, 15, 0))
            )
            rep.modify.material(uv_color)

        # Randomize the camera placement
        cam_seed = 500
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
