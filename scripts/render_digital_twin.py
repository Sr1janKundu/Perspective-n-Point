import os, json, glob, numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, Gf

USD_PATH = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_test02.usd"
PNP_JSON = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\observation\pnp_result.json"
OUTPUT_DIR = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\digital_twin"

CAMERA_PATH = "/World/PnPCamera"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

def opencv_rotation_to_isaac_quaternion(R_opencv):
    S = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    R_isaac_world_to_camera = S @ R_opencv
    R_camera_to_world = R_isaac_world_to_camera.T
    q = Gf.Matrix3d(R_camera_to_world.tolist()).GetQuat()
    return q

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PNP_JSON, "r", encoding="utf-8") as f: pnp = json.load(f)

    camera_position = np.array(pnp["camera_position_world"], dtype=np.float64)
    R_opencv = np.array(pnp["R_world_to_camera_opencv"], dtype=np.float64)
    q = opencv_rotation_to_isaac_quaternion(R_opencv)

    omni.usd.get_context().open_stage(USD_PATH)

    while omni.usd.get_context().get_stage() is None:
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if camera_prim.IsValid(): stage.RemovePrim(CAMERA_PATH)

    camera_prim = stage.DefinePrim(CAMERA_PATH, "Camera")
    camera = UsdGeom.Camera(camera_prim)

    camera.GetFocalLengthAttr().Set(18.14756)
    camera.GetHorizontalApertureAttr().Set(20.9549999)
    camera.GetVerticalApertureAttr().Set(15.29080009)

    xform = UsdGeom.XformCommonAPI(camera_prim)
    xform.SetTranslate(Gf.Vec3d(*camera_position))
    xform.SetOrient(Gf.Quatd(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary())))

    print("PnP camera position:", camera_position)
    print("Creating digital-twin camera at that pose.")

    render_product = rep.create.render_product(CAMERA_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=OUTPUT_DIR, rgb=True)
    writer.attach(render_product)

    rep.orchestrator.set_capture_on_play(False)

    for _ in range(60): simulation_app.update()

    print("Capturing digital-twin image...")
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()

    pngs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    if not pngs: raise RuntimeError("No digital-twin image generated.")

    image_path = os.path.join(OUTPUT_DIR, "digital_twin.png")
    if os.path.abspath(pngs[0]) != os.path.abspath(image_path):
        if os.path.exists(image_path): os.remove(image_path)
        os.replace(pngs[0], image_path)

    print("Saved:", image_path)

finally:
    try: writer.detach()
    except: pass
    try: render_product.destroy()
    except: pass
    simulation_app.close()