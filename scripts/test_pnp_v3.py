import os
import json
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Test OpenCV PnP against Isaac Sim ground truth.")
parser.add_argument("--json", required=True, help="Path to ground_truth.json")
parser.add_argument("--method", default="ITERATIVE", choices=["ITERATIVE", "EPNP", "P3P", "AP3P", "SQPNP"], help="OpenCV solvePnP method")
args = parser.parse_args()

METHODS = {"ITERATIVE": cv2.SOLVEPNP_ITERATIVE, "EPNP": cv2.SOLVEPNP_EPNP, "P3P": cv2.SOLVEPNP_P3P, "AP3P": cv2.SOLVEPNP_AP3P, "SQPNP": cv2.SOLVEPNP_SQPNP}
solvepnp_flag = METHODS[args.method]

with open(args.json, "r", encoding="utf-8") as f:
    dataset = json.load(f)

def rotation_error_degrees(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    trace = np.trace(R_delta)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def camera_position_from_pnp(R, tvec):
    return -R.T @ tvec.reshape(3)

def compute_reprojection_error(object_points, image_points, rvec, tvec, K, dist_coeffs):
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = projected - image_points
    per_point_error = np.linalg.norm(errors, axis=1)
    rms = np.sqrt(np.mean(per_point_error ** 2))
    return rms, per_point_error, projected

def fmt_vector(v):
    return "[" + ", ".join(f"{x:.9f}" for x in np.asarray(v).reshape(-1)) + "]"

cameras = dataset["cameras"]
fiducials = dataset["fiducials"]
fiducial_paths = list(fiducials.keys())

print()
print("=" * 80)
print("PnP VALIDATION")
print("=" * 80)
print(f"Dataset: {args.json}")
print(f"Method: {args.method}")
print(f"Fiducials: {len(fiducial_paths)}")
print(f"Cameras: {len(cameras)}")
print("=" * 80)

all_position_errors = []
all_rotation_errors = []
all_reprojection_errors = []
pnp_results = {"metadata": {"ground_truth_file": args.json, "method": args.method, "fiducials": len(fiducial_paths), "cameras": len(cameras)}, "cameras": {}}

for camera_path, camera_data in cameras.items():
    print()
    print("=" * 80)
    print(f"CAMERA: {camera_path}")
    print("=" * 80)

    K = np.array(camera_data["intrinsics"]["K"], dtype=np.float64)
    distortion = np.array(camera_data["intrinsics"]["distortion_coefficients"], dtype=np.float64)
    object_points = np.array([camera_data["correspondences"][path]["point_3d_world"] for path in fiducial_paths], dtype=np.float64)
    image_points = np.array([camera_data["correspondences"][path]["point_2d_pixel"] for path in fiducial_paths], dtype=np.float64)
    gt_position = np.array(camera_data["world_pose"]["position"], dtype=np.float64)
    R_gt = np.array(camera_data["opencv_extrinsics"]["R_world_to_camera"], dtype=np.float64)
    t_gt = np.array(camera_data["opencv_extrinsics"]["t_world_to_camera"], dtype=np.float64)

    if np.any(~np.isfinite(object_points)):
        raise RuntimeError(f"Invalid 3D points in {camera_path}")
    if np.any(~np.isfinite(image_points)):
        raise RuntimeError(f"Invalid 2D points in {camera_path}")

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, distortion, flags=solvepnp_flag)

    if not success:
        print("PnP FAILED")
        pnp_results["cameras"][camera_path] = {"success": False}
        continue

    R_est, _ = cv2.Rodrigues(rvec)
    estimated_position = camera_position_from_pnp(R_est, tvec)
    position_error = np.linalg.norm(estimated_position - gt_position)
    rotation_error = rotation_error_degrees(R_est, R_gt)
    reprojection_rms, per_point_error, projected_points = compute_reprojection_error(object_points, image_points, rvec, tvec, K, distortion)
    gt_position_from_Rt = -R_gt.T @ t_gt
    gt_consistency_error = np.linalg.norm(gt_position - gt_position_from_Rt)

    all_position_errors.append(position_error)
    all_rotation_errors.append(rotation_error)
    all_reprojection_errors.append(reprojection_rms)

    pnp_results["cameras"][camera_path] = {
        "success": True,
        "ground_truth": {
            "camera_position_world": gt_position.tolist(),
            "R_world_to_camera": R_gt.tolist(),
            "t_world_to_camera": t_gt.tolist()
        },
        "estimated": {
            "camera_position_world": estimated_position.tolist(),
            "R_world_to_camera": R_est.tolist(),
            "rvec": rvec.reshape(3).tolist(),
            "tvec": tvec.reshape(3).tolist()
        },
        "errors": {
            "position_error_m": float(position_error),
            "rotation_error_degrees": float(rotation_error),
            "reprojection_rms_pixels": float(reprojection_rms),
            "ground_truth_position_consistency_error_m": float(gt_consistency_error)
        },
        "per_point_reprojection_errors_pixels": {
            path: float(per_point_error[i]) for i, path in enumerate(fiducial_paths)
        },
        "projected_points": {
            path: projected_points[i].tolist() for i, path in enumerate(fiducial_paths)
        }
    }

    print()
    print("Camera intrinsics:")
    print(K)
    print()
    print("Number of correspondences:", len(object_points))
    print()
    print("Ground-truth camera position:")
    print(fmt_vector(gt_position))
    print()
    print("Estimated camera position:")
    print(fmt_vector(estimated_position))
    print()
    print("Position error:")
    print(f"{position_error:.12f} m")
    print()
    print("Rotation error:")
    print(f"{rotation_error:.12f} degrees")
    print()
    print("Reprojection RMS:")
    print(f"{reprojection_rms:.12f} pixels")
    print()
    print("Ground-truth position consistency:")
    print(f"{gt_consistency_error:.12f} m")
    print()
    print("Estimated rvec:")
    print(rvec.reshape(3))
    print()
    print("Estimated tvec:")
    print(tvec.reshape(3))
    print()
    print("Ground-truth R:")
    print(R_gt)
    print()
    print("Estimated R:")
    print(R_est)
    print()
    print("Per-point reprojection errors:")
    for i, path in enumerate(fiducial_paths):
        print(f"  {path:20s} {per_point_error[i]:.12f} px")

results_dir = os.path.dirname(os.path.abspath(args.json))
results_path = os.path.join(results_dir, "pnp_results.json")

if len(all_position_errors) > 0:
    pnp_results["summary"] = {
        "successful_cameras": len(all_position_errors),
        "position_error_mean_m": float(np.mean(all_position_errors)),
        "position_error_max_m": float(np.max(all_position_errors)),
        "rotation_error_mean_degrees": float(np.mean(all_rotation_errors)),
        "rotation_error_max_degrees": float(np.max(all_rotation_errors)),
        "reprojection_rms_mean_pixels": float(np.mean(all_reprojection_errors)),
        "reprojection_rms_max_pixels": float(np.max(all_reprojection_errors))
    }
else:
    pnp_results["summary"] = {"successful_cameras": 0}

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(pnp_results, f, indent=4, allow_nan=False)

print()
print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

if len(all_position_errors) > 0:
    print(f"Position error (mean): {np.mean(all_position_errors):.12e} m")
    print(f"Position error (max):  {np.max(all_position_errors):.12e} m")
    print(f"Rotation error (mean): {np.mean(all_rotation_errors):.12e} deg")
    print(f"Rotation error (max):  {np.max(all_rotation_errors):.12e} deg")
    print(f"Reprojection RMS (mean): {np.mean(all_reprojection_errors):.12e} px")
    print(f"Reprojection RMS (max):  {np.max(all_reprojection_errors):.12e} px")
else:
    print("No camera produced a successful PnP solution.")

print()
print(f"Results saved to: {results_path}")
print("=" * 80)