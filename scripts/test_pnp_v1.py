import os
import json
import argparse

import cv2
import numpy as np


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Test OpenCV PnP against Isaac Sim ground truth."
)

parser.add_argument(
    "--json",
    required=True,
    help="Path to ground_truth.json"
)

parser.add_argument(
    "--method",
    default="ITERATIVE",
    choices=[
        "ITERATIVE",
        "EPNP",
        "P3P",
        "AP3P",
        "SQPNP"
    ],
    help="OpenCV solvePnP method"
)

args = parser.parse_args()


# ============================================================
# SOLVEPNP METHOD
# ============================================================

METHODS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "EPNP": cv2.SOLVEPNP_EPNP,
    "P3P": cv2.SOLVEPNP_P3P,
    "AP3P": cv2.SOLVEPNP_AP3P,
    "SQPNP": cv2.SOLVEPNP_SQPNP,
}

solvepnp_flag = METHODS[args.method]


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    args.json,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def rotation_error_degrees(R_est, R_gt):
    """
    Angular difference between two rotation matrices.
    """

    R_delta = R_est @ R_gt.T

    trace = np.trace(R_delta)

    cos_angle = (
        (trace - 1.0) / 2.0
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cos_angle)
    )


def camera_position_from_pnp(
    R,
    tvec
):
    """
    PnP gives:

        X_camera = R X_world + t

    Camera center in world coordinates is:

        C = -R^T t
    """

    return (
        -R.T @ tvec.reshape(3)
    )


def compute_reprojection_error(
    object_points,
    image_points,
    rvec,
    tvec,
    K,
    dist_coeffs
):
    """
    Reproject 3D points using estimated PnP pose and calculate
    RMS pixel error.
    """

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist_coeffs
    )

    projected = projected.reshape(
        -1,
        2
    )

    errors = (
        projected
        - image_points
    )

    per_point_error = np.linalg.norm(
        errors,
        axis=1
    )

    rms = np.sqrt(
        np.mean(
            per_point_error ** 2
        )
    )

    return (
        rms,
        per_point_error,
        projected
    )


def fmt_vector(v):
    return "[" + ", ".join(
        f"{x:.9f}"
        for x in v
    ) + "]"


# ============================================================
# DATASET INFO
# ============================================================

cameras = dataset["cameras"]
fiducials = dataset["fiducials"]

fiducial_paths = list(
    fiducials.keys()
)


# ============================================================
# PROCESS EACH CAMERA
# ============================================================

print()
print("=" * 80)
print("PnP VALIDATION")
print("=" * 80)

print(
    f"Dataset: {args.json}"
)

print(
    f"Method: {args.method}"
)

print(
    f"Fiducials: {len(fiducial_paths)}"
)

print(
    f"Cameras: {len(cameras)}"
)

print("=" * 80)


all_position_errors = []
all_rotation_errors = []
all_reprojection_errors = []


for camera_path, camera_data in cameras.items():

    print()
    print("=" * 80)
    print(
        f"CAMERA: {camera_path}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    K = np.array(
        camera_data[
            "intrinsics"
        ]["K"],
        dtype=np.float64
    )

    distortion = np.array(
        camera_data[
            "intrinsics"
        ][
            "distortion_coefficients"
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # 3D world points
    # --------------------------------------------------------

    object_points = np.array(
        [
            camera_data[
                "correspondences"
            ][path][
                "point_3d_world"
            ]
            for path in fiducial_paths
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # 2D image points
    # --------------------------------------------------------

    image_points = np.array(
        [
            camera_data[
                "correspondences"
            ][path][
                "point_2d_pixel"
            ]
            for path in fiducial_paths
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Ground-truth camera position
    # --------------------------------------------------------

    gt_position = np.array(
        camera_data[
            "world_pose"
        ]["position"],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Ground-truth OpenCV rotation
    # --------------------------------------------------------

    R_gt = np.array(
        camera_data[
            "opencv_extrinsics"
        ][
            "R_world_to_camera"
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Ground-truth OpenCV translation
    # --------------------------------------------------------

    t_gt = np.array(
        camera_data[
            "opencv_extrinsics"
        ][
            "t_world_to_camera"
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Check that all points are valid
    # --------------------------------------------------------

    if np.any(
        ~np.isfinite(object_points)
    ):
        raise RuntimeError(
            f"Invalid 3D points in {camera_path}"
        )

    if np.any(
        ~np.isfinite(image_points)
    ):
        raise RuntimeError(
            f"Invalid 2D points in {camera_path}"
        )

    # --------------------------------------------------------
    # Run solvePnP
    # --------------------------------------------------------

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        distortion,
        flags=solvepnp_flag
    )

    if not success:

        print(
            "PnP FAILED"
        )

        continue

    # --------------------------------------------------------
    # Convert rotation vector -> matrix
    # --------------------------------------------------------

    R_est, _ = cv2.Rodrigues(
        rvec
    )

    # --------------------------------------------------------
    # Recover camera center
    # --------------------------------------------------------

    estimated_position = (
        camera_position_from_pnp(
            R_est,
            tvec
        )
    )

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    position_error = np.linalg.norm(
        estimated_position
        - gt_position
    )

    # --------------------------------------------------------
    # Rotation error
    # --------------------------------------------------------

    rotation_error = (
        rotation_error_degrees(
            R_est,
            R_gt
        )
    )

    # --------------------------------------------------------
    # Reprojection error
    # --------------------------------------------------------

    (
        reprojection_rms,
        per_point_error,
        projected_points
    ) = compute_reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        K,
        distortion
    )

    # --------------------------------------------------------
    # Ground truth camera center calculated from R,t
    # --------------------------------------------------------

    gt_position_from_Rt = (
        -R_gt.T @ t_gt
    )

    gt_consistency_error = np.linalg.norm(
        gt_position
        - gt_position_from_Rt
    )

    # --------------------------------------------------------
    # Store global statistics
    # --------------------------------------------------------

    all_position_errors.append(
        position_error
    )

    all_rotation_errors.append(
        rotation_error
    )

    all_reprojection_errors.append(
        reprojection_rms
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("Camera intrinsics:")
    print(K)

    print()
    print(
        "Number of correspondences:",
        len(object_points)
    )

    print()
    print(
        "Ground-truth camera position:"
    )

    print(
        fmt_vector(gt_position)
    )

    print()
    print(
        "Estimated camera position:"
    )

    print(
        fmt_vector(estimated_position)
    )

    print()
    print(
        "Position error:"
    )

    print(
        f"{position_error:.12f} m"
    )

    print()
    print(
        "Rotation error:"
    )

    print(
        f"{rotation_error:.12f} degrees"
    )

    print()
    print(
        "Reprojection RMS:"
    )

    print(
        f"{reprojection_rms:.12f} pixels"
    )

    print()
    print(
        "Ground-truth position consistency:"
    )

    print(
        f"{gt_consistency_error:.12f} m"
    )

    print()
    print(
        "Estimated rvec:"
    )

    print(
        rvec.reshape(3)
    )

    print()
    print(
        "Estimated tvec:"
    )

    print(
        tvec.reshape(3)
    )

    print()
    print(
        "Ground-truth R:"
    )

    print(
        R_gt
    )

    print()
    print(
        "Estimated R:"
    )

    print(
        R_est
    )

    # --------------------------------------------------------
    # Individual point errors
    # --------------------------------------------------------

    print()
    print(
        "Per-point reprojection errors:"
    )

    for i, path in enumerate(
        fiducial_paths
    ):

        print(
            f"  {path:20s} "
            f"{per_point_error[i]:.12f} px"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

if len(all_position_errors) > 0:

    print(
        f"Position error "
        f"(mean): "
        f"{np.mean(all_position_errors):.12e} m"
    )

    print(
        f"Position error "
        f"(max):  "
        f"{np.max(all_position_errors):.12e} m"
    )

    print(
        f"Rotation error "
        f"(mean): "
        f"{np.mean(all_rotation_errors):.12e} deg"
    )

    print(
        f"Rotation error "
        f"(max):  "
        f"{np.max(all_rotation_errors):.12e} deg"
    )

    print(
        f"Reprojection RMS "
        f"(mean): "
        f"{np.mean(all_reprojection_errors):.12e} px"
    )

    print(
        f"Reprojection RMS "
        f"(max):  "
        f"{np.max(all_reprojection_errors):.12e} px"
    )

else:

    print(
        "No camera produced a successful PnP solution."
    )

print("=" * 80)