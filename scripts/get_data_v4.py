import os
import time
import json
import glob
import math
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
import isaacsim.core.experimental.utils.xform as xform_utils

USD_PATH = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_test02.usd"
OUTPUT_DIR = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\test02"

# FIDUCIAL_PATHS = ["/World/ref01", "/World/ref01_01", "/World/ref01_02", "/World/ref01_03", "/World/ref01_04", "/World/ref01_05", "/World/ref01_06", "/World/ref01_07"]
FIDUCIAL_PATHS = [f"/World/Fiducials_{i:02d}" for i in range(1, 14)]
CAMERA_PATHS = ["/World/Camera", "/World/Camera_01", "/World/Camera_02", "/World/Camera_03"]

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
RESOLUTION = (IMAGE_WIDTH, IMAGE_HEIGHT)
RENDER_WARMUP_FRAMES = 60

def warp_to_numpy(value):
    if hasattr(value, "numpy"):
        return np.array(value.numpy(), dtype=np.float64)
    return np.array(value, dtype=np.float64)

def get_world_pose_numpy(prim):
    position, orientation = xform_utils.get_world_pose(prim)
    position = warp_to_numpy(position)
    orientation = warp_to_numpy(orientation)
    return position, orientation

def quaternion_wxyz_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    R = np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float64)
    return R

def get_attr(prim, name, default=None):
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        return default
    value = attr.Get()
    if value is None:
        return default
    return value

def get_camera_intrinsics(prim, width, height):
    focal_length = float(get_attr(prim, "focalLength"))
    horizontal_aperture = float(get_attr(prim, "horizontalAperture"))
    vertical_aperture = float(get_attr(prim, "verticalAperture"))
    horizontal_offset = float(get_attr(prim, "horizontalApertureOffset", 0.0))
    vertical_offset = float(get_attr(prim, "verticalApertureOffset", 0.0))
    projection = str(get_attr(prim, "projection", "perspective"))
    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    horizontal_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
    vertical_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
    clipping_range = get_attr(prim, "clippingRange", None)
    if clipping_range is not None:
        clipping_range = [float(clipping_range[0]), float(clipping_range[1])]
    return {"projection": projection, "image_width": width, "image_height": height, "focal_length": focal_length, "horizontal_aperture": horizontal_aperture, "vertical_aperture": vertical_aperture, "horizontal_aperture_offset": horizontal_offset, "vertical_aperture_offset": vertical_offset, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K.tolist(), "distortion_model": "none", "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0], "horizontal_fov_degrees": math.degrees(horizontal_fov), "vertical_fov_degrees": math.degrees(vertical_fov), "clipping_range": clipping_range}

def get_opencv_extrinsics(camera_position, camera_orientation):
    R_camera_to_world_isaac = quaternion_wxyz_to_rotation_matrix(camera_orientation)
    R_world_to_camera_isaac = R_camera_to_world_isaac.T
    S = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    R_world_to_camera_opencv = S @ R_world_to_camera_isaac
    C_world = np.asarray(camera_position, dtype=np.float64).reshape(3)
    t_world_to_camera_opencv = -R_world_to_camera_opencv @ C_world
    T_world_to_camera = np.eye(4, dtype=np.float64)
    T_world_to_camera[:3, :3] = R_world_to_camera_opencv
    T_world_to_camera[:3, 3] = t_world_to_camera_opencv
    C_recovered = -R_world_to_camera_opencv.T @ t_world_to_camera_opencv
    return R_world_to_camera_opencv, t_world_to_camera_opencv, T_world_to_camera, C_recovered

def project_points(points_world, K, R, t):
    points_world = np.asarray(points_world, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    points_camera = (R @ points_world.T).T + t
    projected_points = []
    for point in points_camera:
        X, Y, Z = point
        if Z <= 0:
            projected_points.append(None)
            continue
        u = K[0, 0] * X / Z + K[0, 2]
        v = K[1, 1] * Y / Z + K[1, 2]
        projected_points.append([float(u), float(v)])
    return projected_points, points_camera

def find_png(directory):
    pngs = glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)
    if len(pngs) == 0:
        return None
    return pngs[0]

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Loading USD:")
    print(USD_PATH)
    print("=" * 70)
    omni.usd.get_context().open_stage(USD_PATH)
    load_start = time.time()

    while True:
        simulation_app.update()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            continue
        missing_fiducials = [path for path in FIDUCIAL_PATHS if not stage.GetPrimAtPath(path).IsValid()]
        missing_cameras = [path for path in CAMERA_PATHS if not stage.GetPrimAtPath(path).IsValid()]
        if len(missing_fiducials) == 0 and len(missing_cameras) == 0:
            break

    load_time = time.time() - load_start
    print()
    print("=" * 70)
    print("SCENE READY")
    print(f"Scene readiness time: {load_time:.2f} seconds")
    print("=" * 70)

    print("\n========== VALIDATING FIDUCIALS ==========")
    fiducial_data = {}
    for path in FIDUCIAL_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Fiducial not found: {path}")
        position, orientation = get_world_pose_numpy(prim)
        fiducial_data[path] = {"position_world": position.tolist(), "orientation_wxyz": orientation.tolist()}
        print(f"{path}: {position}")

    print("\n========== CAMERA PARAMETERS ==========")
    camera_data = {}
    for camera_path in CAMERA_PATHS:
        prim = stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            raise RuntimeError(f"Camera not found: {camera_path}")
        if prim.GetTypeName() != "Camera":
            raise RuntimeError(f"{camera_path} is not a Camera. Type = {prim.GetTypeName()}")

        position, orientation = get_world_pose_numpy(prim)
        intrinsics = get_camera_intrinsics(prim, IMAGE_WIDTH, IMAGE_HEIGHT)
        R_wc, t_wc, T_wc, recovered_center = get_opencv_extrinsics(position, orientation)

        camera_data[camera_path] = {"world_pose": {"position": position.tolist(), "orientation_wxyz": orientation.tolist()}, "intrinsics": intrinsics, "opencv_extrinsics": {"R_world_to_camera": R_wc.tolist(), "t_world_to_camera": t_wc.tolist(), "T_world_to_camera": T_wc.tolist(), "camera_center_world": recovered_center.tolist()}}

        print()
        print("Camera:", camera_path)
        print("Position:", position)
        print("Orientation (wxyz):", orientation)
        print("Focal length:", intrinsics["focal_length"])
        print("Horizontal aperture:", intrinsics["horizontal_aperture"])
        print("Vertical aperture:", intrinsics["vertical_aperture"])
        print("K:")
        print(np.array(intrinsics["K"]))

    print("\n========== GENERATING 2D GROUND TRUTH ==========")
    fiducial_positions = np.array([fiducial_data[path]["position_world"] for path in FIDUCIAL_PATHS], dtype=np.float64)

    for camera_path in CAMERA_PATHS:
        camera = camera_data[camera_path]
        K = np.array(camera["intrinsics"]["K"], dtype=np.float64)
        R = np.array(camera["opencv_extrinsics"]["R_world_to_camera"], dtype=np.float64)
        t = np.array(camera["opencv_extrinsics"]["t_world_to_camera"], dtype=np.float64)
        projected_points, camera_points = project_points(fiducial_positions, K, R, t)
        camera["correspondences"] = {}
        for i, path in enumerate(FIDUCIAL_PATHS):
            camera["correspondences"][path] = {"point_3d_world": fiducial_positions[i].tolist(), "point_3d_camera_opencv": camera_points[i].tolist(), "point_2d_pixel": projected_points[i]}

    print("\n========== CREATING RENDER PRODUCTS ==========")
    render_products = []
    for camera_path in CAMERA_PATHS:
        camera_name = camera_path.split("/")[-1]
        print(f"Creating render product: {camera_path}")
        render_product = rep.create.render_product(camera_path, RESOLUTION)
        render_products.append((camera_path, camera_name, render_product))

    print("\n========== SETTING UP WRITERS ==========")
    writers = []
    for camera_path, camera_name, render_product in render_products:
        camera_output_dir = os.path.join(OUTPUT_DIR, f"camera_{camera_name}")
        os.makedirs(camera_output_dir, exist_ok=True)
        writer = rep.writers.get("BasicWriter")
        writer.initialize(output_dir=camera_output_dir, rgb=True)
        writer.attach(render_product)
        writers.append((camera_path, camera_name, writer, camera_output_dir))
        print(f"{camera_name} -> {camera_output_dir}")

    print("\n========== RENDERER WARM-UP ==========")
    rep.orchestrator.set_capture_on_play(False)
    for i in range(RENDER_WARMUP_FRAMES):
        simulation_app.update()
        if i % 10 == 0:
            print(f"Renderer warm-up: {i}/{RENDER_WARMUP_FRAMES}")

    print("\n========== CAPTURING ==========")
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    print("Capture complete.")

    print("\n========== COLLECTING IMAGES ==========")
    image_paths = {}
    for camera_path, camera_name, writer, camera_output_dir in writers:
        generated_png = find_png(camera_output_dir)
        if generated_png is None:
            raise RuntimeError(f"No PNG was generated for {camera_path}")
        final_path = os.path.join(camera_output_dir, "image.png")
        if os.path.abspath(generated_png) != os.path.abspath(final_path):
            if os.path.exists(final_path):
                os.remove(final_path)
            os.replace(generated_png, final_path)
        image_paths[camera_path] = os.path.relpath(final_path, OUTPUT_DIR)
        print(f"{camera_path} -> {final_path}")

    for camera_path in CAMERA_PATHS:
        camera_data[camera_path]["image"] = image_paths[camera_path]

    dataset = {"metadata": {"description": "Isaac Sim ground-truth dataset for PnP camera pose evaluation", "usd_file": USD_PATH, "image_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT], "coordinate_systems": {"isaac_world": "Isaac Sim world coordinates", "isaac_camera": "+X right, +Y up, -Z forward", "opencv_camera": "+X right, +Y down, +Z forward"}, "intrinsic_model": "Perspective pinhole", "distortion": "Zero distortion"}, "fiducials": fiducial_data, "cameras": camera_data}

    json_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    print()
    print("========== SAVING GROUND TRUTH ==========")
    print(f"Saving: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, allow_nan=False)
    print("ground_truth.json saved successfully.")

    print("\n========== VALIDATING JSON ==========")
    with open(json_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)
    print("JSON validation successful.")
    print(f"Fiducials: {len(test_dataset['fiducials'])}")
    print(f"Cameras: {len(test_dataset['cameras'])}")

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)
    print(f"Output directory:\n{OUTPUT_DIR}")
    print(f"\nGround truth:\n{json_path}")
    print()

    for camera_path in CAMERA_PATHS:
        camera = test_dataset["cameras"][camera_path]
        visible_points = sum(1 for correspondence in camera["correspondences"].values() if correspondence["point_2d_pixel"] is not None)
        print(camera_path)
        print(f"  Image: {camera['image']}")
        print(f"  Camera center: {camera['world_pose']['position']}")
        print(f"  fx: {camera['intrinsics']['fx']:.6f}")
        print(f"  fy: {camera['intrinsics']['fy']:.6f}")
        print(f"  cx: {camera['intrinsics']['cx']:.6f}")
        print(f"  cy: {camera['intrinsics']['cy']:.6f}")
        print(f"  Visible 3D points: {visible_points}/{len(FIDUCIAL_PATHS)}")
        print()

finally:
    try:
        if "writers" in locals():
            for camera_path, camera_name, writer, camera_output_dir in writers:
                try:
                    writer.detach()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if "render_products" in locals():
            for camera_path, camera_name, render_product in render_products:
                try:
                    render_product.destroy()
                except Exception:
                    pass
    except Exception:
        pass

    simulation_app.close()
    print("\nDone.")
