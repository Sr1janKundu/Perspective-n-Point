# run with `uv run .\scripts\solve_pnp.py --json "C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\observation\observation.json"`

import json, argparse, cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
args = parser.parse_args()

with open(args.json, "r", encoding="utf-8") as f: data = json.load(f)

camera = data["camera"]
K = np.array(camera["intrinsics"]["K"], dtype=np.float64)
dist = np.zeros((5,1), dtype=np.float64)
fiducial_paths = list(camera["correspondences"].keys())
object_points = np.array([camera["correspondences"][p]["point_3d_world"] for p in fiducial_paths], dtype=np.float64)
image_points = np.array([camera["correspondences"][p]["point_2d_pixel"] for p in fiducial_paths], dtype=np.float64)

success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
if not success: raise RuntimeError("solvePnP failed")

R_est, _ = cv2.Rodrigues(rvec)
camera_position = -R_est.T @ tvec.reshape(3)

result = {"source_observation": args.json, "camera_position_world": camera_position.tolist(), "R_world_to_camera_opencv": R_est.tolist(), "rvec": rvec.reshape(3).tolist(), "tvec": tvec.reshape(3).tolist(), "K": K.tolist()}

output = args.json.replace("observation.json", "pnp_result.json")
with open(output, "w", encoding="utf-8") as f: json.dump(result, f, indent=4)

print("PnP camera position:", camera_position)
print("PnP R:\n", R_est)
print("PnP t:", tvec.reshape(3))
print("Saved:", output)