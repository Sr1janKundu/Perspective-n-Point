import os
import time
from isaacsim import SimulationApp

# ------------------------------------------------------------
# Isaac Sim configuration
# ------------------------------------------------------------

simulation_app = SimulationApp({
    "headless": False
})

import omni.usd
import omni.replicator.core as rep
import isaacsim.core.experimental.utils.xform as xform_utils


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

USD_PATH = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_test01.usd"

OUTPUT_DIR = r"C:\Users\sr1ja\Downloads\PnP_test\camera captures"

# Fiducial paths
Fiducial_PATHS = [
    "/World/ref01",
    "/World/ref01_01",
    "/World/ref01_02",
    "/World/ref01_03",
    "/World/ref01_04",
    "/World/ref01_05",
    "/World/ref01_06",
    "/World/ref01_07",
]

# Existing camera prims
CAMERA_PATHS = [
    "/World/Camera",
    "/World/Camera_01",
    "/World/Camera_02",
    "/World/Camera_03",
]

# Image resolution
RESOLUTION = (1024, 1024)

# Minimum time to allow the scene to settle
MIN_LOAD_TIME = 300.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load USD
# ------------------------------------------------------------

print("=" * 70)
print(f"Loading USD:")
print(USD_PATH)
print("=" * 70)

omni.usd.get_context().open_stage(USD_PATH)

load_start = time.time()
last_report = -10

while True:

    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    elapsed = time.time() - load_start

    if stage is None:
        ready = False
        missing_fiducials = Fiducial_PATHS
        missing_cameras = CAMERA_PATHS

    else:

        missing_fiducials = [
            path
            for path in Fiducial_PATHS
            if not stage.GetPrimAtPath(path).IsValid()
        ]

        missing_cameras = [
            path
            for path in CAMERA_PATHS
            if not stage.GetPrimAtPath(path).IsValid()
        ]

        ready = (
            len(missing_fiducials) == 0
            and len(missing_cameras) == 0
        )

    # Print status every 10 seconds
    if elapsed - last_report >= 10:

        print("\n" + "-" * 70)
        print(f"Elapsed: {elapsed:.1f} seconds")
        print(f"Stage exists: {stage is not None}")

        print("Missing fiducials:")
        if missing_fiducials:
            for path in missing_fiducials:
                print(f"    {path}")
        else:
            print("    NONE")

        print("Missing cameras:")
        if missing_cameras:
            for path in missing_cameras:
                print(f"    {path}")
        else:
            print("    NONE")

        print("-" * 70)

        last_report = elapsed

    # Require BOTH:
    # 1. At least 300 seconds have passed
    # 2. All required Prims exist

    if elapsed >= MIN_LOAD_TIME and ready:
        break


print("\n" + "=" * 70)
print("SCENE READY")
print(f"Total wait time: {time.time() - load_start:.1f} seconds")
print("=" * 70)


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError("Failed to obtain USD stage.")

print("\n========== VALIDATING FIDUCIALS ==========")

for object_path in Fiducial_PATHS:

    prim = stage.GetPrimAtPath(object_path)

    if not prim.IsValid():
        raise RuntimeError(
            f"Expected fiducial was not found: {object_path}"
        )

    print(f"[OK] {object_path}")


print("\n========== VALIDATING CAMERAS ==========")

for camera_path in CAMERA_PATHS:

    prim = stage.GetPrimAtPath(camera_path)

    if not prim.IsValid():
        raise RuntimeError(
            f"Expected camera was not found: {camera_path}"
        )

    if prim.GetTypeName() != "Camera":
        raise RuntimeError(
            f"{camera_path} exists but is not a Camera. "
            f"Type: {prim.GetTypeName()}"
        )

    print(f"[OK] {camera_path}")


# ------------------------------------------------------------
# 1. Get object positions
# ------------------------------------------------------------

print("\n========== OBJECT POSITIONS ==========")

for object_path in Fiducial_PATHS:

    prim = stage.GetPrimAtPath(object_path)

    position, orientation = xform_utils.get_world_pose(prim)

    print(f"\nObject: {object_path}")
    print(f"Position: {position}")
    print(f"Orientation (wxyz): {orientation}")


# ------------------------------------------------------------
# 2. Create render products
# ------------------------------------------------------------

print("\n========== CREATING CAMERAS ==========")

render_products = []

for camera_path in CAMERA_PATHS:

    print(f"Creating render product: {camera_path}")

    render_product = rep.create.render_product(
        camera_path,
        RESOLUTION
    )

    render_products.append(
        (camera_path, render_product)
    )


# ------------------------------------------------------------
# 3. Set up writer
# ------------------------------------------------------------

print("\n========== SETTING UP WRITER ==========")

writer = rep.writers.get("BasicWriter")

writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True
)

writer.attach(
    [rp for _, rp in render_products]
)


# ------------------------------------------------------------
# 4. Give renderer additional time to initialize
# ------------------------------------------------------------

print("\n========== INITIALIZING RENDERER ==========")

for i in range(60):

    simulation_app.update()

    if i % 10 == 0:
        print(f"Renderer warm-up: {i}/60 frames")


# ------------------------------------------------------------
# 5. Capture one frame
# ------------------------------------------------------------

print("\n========== CAPTURING ==========")

rep.orchestrator.set_capture_on_play(False)

simulation_app.update()

rep.orchestrator.step()

# Allow asynchronous writing/rendering to finish
for _ in range(30):
    simulation_app.update()


print("\nCapture complete.")


# ------------------------------------------------------------
# 6. Print camera positions
# ------------------------------------------------------------

print("\n========== CAMERA POSITIONS ==========")

for camera_path, _ in render_products:

    prim = stage.GetPrimAtPath(camera_path)

    position, orientation = xform_utils.get_world_pose(prim)

    print(f"\nCamera: {camera_path}")
    print(f"Position: {position}")
    print(f"Orientation (wxyz): {orientation}")


# ------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------

simulation_app.close()

print("\n" + "=" * 70)
print("DONE")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)