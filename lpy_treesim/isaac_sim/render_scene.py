import omni.replicator.core as rep
import omni.usd
        
# 1. Define the scene and assets
# Note: replace the USD paths below with your own 3D assets
tree_path = "/models/lpy_envy_00000.usda"
camera_path = "/World/Camera"
output_directory = "/isaac_sim_output/training_data"

stage_dir = "/home/cindy/isaacsim/World/models/lpy_envy_00000.usda"

usd_context = omni.usd.get_context()
usd_context.open_stage(stage_dir)

with rep.new_layer():
    # Create a simple environment
    rep.create.plane(scale=10, visible=True)
    
    # Spawn the object to be detected
    tree = rep.create.from_usd("lpy_envy_00000")
    
    # Spawn the camera and attach to a render product
    camera = rep.create.camera(position=(0, -5, 2), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, (512, 512))

    # 2. Define domain randomization
    with rep.trigger.on_frame(max_execs=10):
        # Randomize the cube's position and rotation
        with tree:
            rep.modify.pose(
                position=rep.distribution.uniform((-2, -2, 1), (2, 2, 2)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
            )

        # Randomize the camera placement
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-6, -6, 2), (-4, -4, 4)),
                look_at=(0, 0, 0)
            )

    # 3. Set up the writer (saves to disk)
    # This writer saves RGB, Bounding Boxes, and Depth data
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        bounding_box_2d_loose=True
    )
    
    # Attach the writer to the render product and run
    writer.attach([render_product])
    
    # Execute the simulation
    rep.orchestration.run()
