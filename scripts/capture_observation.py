import os
import json
import glob
import math
import traceback
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
import isaacsim.core.experimental.utils.xform as xform_utils
from pxr import UsdGeom, Gf


# ============================================================
# CONFIG
# ============================================================

USD_PATH = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_test02.usd"
OUTPUT_DIR = r"C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\observation"

FIDUCIAL_PATHS = [f"/World/Fiducials_{i:02d}" for i in range(1, 14)]
CAMERA_PATH = "/World/CaptureCamera"

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
RESOLUTION = (IMAGE_WIDTH, IMAGE_HEIGHT)

TARGET = np.array([0.0, 0.0, 2.0], dtype=np.float64)

FOCAL_LENGTH = 18.14756
HORIZONTAL_APERTURE = 20.9549999
VERTICAL_APERTURE = 15.29080009

WARMUP_FRAMES = 60


# ============================================================
# UTILITIES
# ============================================================

def warp_to_numpy(value):
    if hasattr(value, "numpy"):
        return np.array(value.numpy(), dtype=np.float64)
    return np.array(value, dtype=np.float64)


def get_world_pose_numpy(prim):
    position, orientation = xform_utils.get_world_pose(prim)
    return warp_to_numpy(position), warp_to_numpy(orientation)


def quaternion_wxyz_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def rotation_matrix_to_quaternion_wxyz(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def get_camera_intrinsics(prim):
    focal_length = float(prim.GetAttribute("focalLength").Get())
    horizontal_aperture = float(prim.GetAttribute("horizontalAperture").Get())
    vertical_aperture = float(prim.GetAttribute("verticalAperture").Get())

    horizontal_offset_attr = prim.GetAttribute("horizontalApertureOffset")
    vertical_offset_attr = prim.GetAttribute("verticalApertureOffset")

    horizontal_offset = float(horizontal_offset_attr.Get()) if horizontal_offset_attr.IsValid() else 0.0
    vertical_offset = float(vertical_offset_attr.Get()) if vertical_offset_attr.IsValid() else 0.0

    fx = IMAGE_WIDTH * focal_length / horizontal_aperture
    fy = IMAGE_HEIGHT * focal_length / vertical_aperture
    cx = IMAGE_WIDTH / 2.0
    cy = IMAGE_HEIGHT / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    horizontal_fov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
    vertical_fov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))

    return {
        "projection": "perspective",
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "focal_length": focal_length,
        "horizontal_aperture": horizontal_aperture,
        "vertical_aperture": vertical_aperture,
        "horizontal_aperture_offset": horizontal_offset,
        "vertical_aperture_offset": vertical_offset,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K.tolist(),
        "distortion_model": "none",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "horizontal_fov_degrees": math.degrees(horizontal_fov),
        "vertical_fov_degrees": math.degrees(vertical_fov),
        "clipping_range": None
    }


def get_opencv_extrinsics(camera_position, camera_orientation):
    R_camera_to_world_isaac = quaternion_wxyz_to_rotation_matrix(camera_orientation)
    R_world_to_camera_isaac = R_camera_to_world_isaac.T

    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)

    R_world_to_camera_opencv = S @ R_world_to_camera_isaac

    camera_center_world = np.asarray(camera_position, dtype=np.float64).reshape(3)
    t_world_to_camera_opencv = -R_world_to_camera_opencv @ camera_center_world

    T_world_to_camera = np.eye(4, dtype=np.float64)
    T_world_to_camera[:3, :3] = R_world_to_camera_opencv
    T_world_to_camera[:3, 3] = t_world_to_camera_opencv

    recovered_center = -R_world_to_camera_opencv.T @ t_world_to_camera_opencv

    return (
        R_world_to_camera_opencv,
        t_world_to_camera_opencv,
        T_world_to_camera,
        recovered_center
    )


def project_points(points_world, K, R, t):
    points_world = np.asarray(points_world, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    points_camera = (R @ points_world.T).T + t
    projected_points = []

    for X, Y, Z in points_camera:
        if Z <= 0.0:
            projected_points.append(None)
        else:
            u = K[0, 0] * X / Z + K[0, 2]
            v = K[1, 1] * Y / Z + K[1, 2]
            projected_points.append([float(u), float(v)])

    return projected_points, points_camera


# ============================================================
# CAMERA CREATION
# ============================================================

def create_capture_camera(stage):
    print("\n========== CREATING UNKNOWN CAMERA ==========")

    existing_camera = stage.GetPrimAtPath(CAMERA_PATH)
    if existing_camera.IsValid():
        print(f"Removing existing camera: {CAMERA_PATH}")
        stage.RemovePrim(CAMERA_PATH)

    camera_prim = stage.DefinePrim(CAMERA_PATH, "Camera")
    if not camera_prim.IsValid():
        raise RuntimeError(f"Failed to create camera: {CAMERA_PATH}")

    camera = UsdGeom.Camera(camera_prim)
    camera.GetFocalLengthAttr().Set(FOCAL_LENGTH)
    camera.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE)
    camera.GetVerticalApertureAttr().Set(VERTICAL_APERTURE)

    radius = np.random.uniform(4.0, 5.0)

    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)

    position = direction * radius

    forward = TARGET - position
    forward /= np.linalg.norm(forward)

    up_reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if abs(np.dot(forward, up_reference)) > 0.99:
        up_reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, up_reference)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Isaac camera:
    # +X = right
    # +Y = up
    # -Z = forward

    R_camera_to_world = np.column_stack([
        right,
        up,
        -forward
    ])

    q_wxyz = rotation_matrix_to_quaternion_wxyz(R_camera_to_world)

    # AddOrientOp() creates a GfQuatf attribute in this USD setup.
    quat = Gf.Quatf(
        float(q_wxyz[0]),
        Gf.Vec3f(
            float(q_wxyz[1]),
            float(q_wxyz[2]),
            float(q_wxyz[3])
        )
    )

    xformable = UsdGeom.Xformable(camera_prim)

    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()

    translate_op.Set(
        Gf.Vec3d(
            float(position[0]),
            float(position[1]),
            float(position[2])
        )
    )

    orient_op.Set(quat)

    print(f"Camera path: {CAMERA_PATH}")
    print(f"Camera position: {position}")
    print(f"Distance from origin: {np.linalg.norm(position):.6f} m")
    print(f"Target: {TARGET}")
    print(f"Forward direction: {forward}")
    print(f"Quaternion WXYZ: {q_wxyz}")

    return camera_prim


# ============================================================
# MAIN
# ============================================================

writer = None
render_product = None

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LOADING USD")
    print("=" * 70)
    print(USD_PATH)

    omni.usd.get_context().open_stage(USD_PATH)

    while True:
        simulation_app.update()
        stage = omni.usd.get_context().get_stage()

        if stage is None:
            continue

        if all(stage.GetPrimAtPath(path).IsValid() for path in FIDUCIAL_PATHS):
            break

    print("\n========== SCENE READY ==========")
    print("\n========== READING FIDUCIALS ==========")

    fiducial_data = {}

    for path in FIDUCIAL_PATHS:
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid():
            raise RuntimeError(f"Fiducial not found: {path}")

        position, orientation = get_world_pose_numpy(prim)

        fiducial_data[path] = {
            "position_world": position.tolist(),
            "orientation_wxyz": orientation.tolist()
        }

        print(f"{path}: {position}")

    camera_prim = create_capture_camera(stage)

    for _ in range(3):
        simulation_app.update()

    camera_position, camera_orientation = get_world_pose_numpy(camera_prim)

    print("\n========== CAMERA POSE ==========")
    print("World position:")
    print(camera_position)
    print("World orientation WXYZ:")
    print(camera_orientation)

    intrinsics = get_camera_intrinsics(camera_prim)

    print("\n========== CAMERA INTRINSICS ==========")
    print(np.array(intrinsics["K"]))

    R_wc, t_wc, T_wc, recovered_center = get_opencv_extrinsics(
        camera_position,
        camera_orientation
    )

    print("\n========== OPENCV EXTRINSICS ==========")
    print("R:")
    print(R_wc)
    print("t:")
    print(t_wc)
    print("Recovered camera center:")
    print(recovered_center)

    print("\n========== GENERATING 2D PNP INPUT ==========")

    fiducial_positions = np.array(
        [fiducial_data[path]["position_world"] for path in FIDUCIAL_PATHS],
        dtype=np.float64
    )

    K = np.array(intrinsics["K"], dtype=np.float64)

    projected_points, camera_points = project_points(
        fiducial_positions,
        K,
        R_wc,
        t_wc
    )

    correspondences = {}

    for i, path in enumerate(FIDUCIAL_PATHS):
        correspondences[path] = {
            "point_3d_world": fiducial_positions[i].tolist(),
            "point_3d_camera_opencv": camera_points[i].tolist(),
            "point_2d_pixel": projected_points[i]
        }

        print(
            f"{path}: "
            f"3D={fiducial_positions[i]} "
            f"2D={projected_points[i]}"
        )

    invalid_points = [
        path for path, value in correspondences.items()
        if value["point_2d_pixel"] is None
    ]

    if invalid_points:
        raise RuntimeError(
            f"Fiducials behind camera: {invalid_points}"
        )

    print("\n========== CREATING RENDER PRODUCT ==========")

    render_product = rep.create.render_product(
        CAMERA_PATH,
        RESOLUTION
    )

    writer = rep.writers.get("BasicWriter")

    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True
    )

    writer.attach(render_product)

    print("\n========== RENDERER WARM-UP ==========")

    rep.orchestrator.set_capture_on_play(False)

    for i in range(WARMUP_FRAMES):
        simulation_app.update()

        if i % 10 == 0:
            print(f"Renderer warm-up: {i}/{WARMUP_FRAMES}")

    print("\n========== CAPTURING OBSERVATION ==========")

    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()

    print("Capture complete.")

    print("\n========== COLLECTING IMAGE ==========")

    pngs = glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "**",
            "*.png"
        ),
        recursive=True
    )

    if not pngs:
        raise RuntimeError(
            f"No PNG generated in {OUTPUT_DIR}"
        )

    image_path = os.path.join(
        OUTPUT_DIR,
        "observation.png"
    )

    generated_png = pngs[0]

    if os.path.abspath(generated_png) != os.path.abspath(image_path):
        if os.path.exists(image_path):
            os.remove(image_path)

        os.replace(
            generated_png,
            image_path
        )

    print(f"Image saved: {image_path}")

    observation = {
        "metadata": {
            "type": "observation",
            "usd_file": USD_PATH,
            "image_resolution": [
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ],
            "coordinate_systems": {
                "isaac_world": "Isaac Sim world coordinates",
                "isaac_camera": "+X right, +Y up, -Z forward",
                "opencv_camera": "+X right, +Y down, +Z forward"
            },
            "intrinsic_model": "Perspective pinhole",
            "distortion": "Zero distortion"
        },
        "fiducials": fiducial_data,
        "camera": {
            "path": CAMERA_PATH,
            "intrinsics": intrinsics,
            "ground_truth_pose": {
                "position_world": camera_position.tolist(),
                "orientation_wxyz": camera_orientation.tolist()
            },
            "opencv_extrinsics": {
                "R_world_to_camera": R_wc.tolist(),
                "t_world_to_camera": t_wc.tolist(),
                "T_world_to_camera": T_wc.tolist(),
                "camera_center_world": recovered_center.tolist()
            },
            "correspondences": correspondences,
            "image": "observation.png"
        }
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "observation.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            observation,
            f,
            indent=4,
            allow_nan=False
        )

    print("\n========== COMPLETE ==========")
    print(f"Image: {image_path}")
    print(f"PnP data: {json_path}")

except Exception:
    print("\n========== ERROR ==========")
    traceback.print_exc()
    raise

finally:
    try:
        if writer is not None:
            writer.detach()
    except Exception:
        pass

    try:
        if render_product is not None:
            render_product.destroy()
    except Exception:
        pass

    simulation_app.close()
    print("\nDone.")