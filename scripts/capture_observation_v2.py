import os,json,glob,math,traceback
import numpy as np
from isaacsim import SimulationApp

simulation_app=SimulationApp({"headless":False})

import omni.usd
import omni.replicator.core as rep
import isaacsim.core.experimental.utils.xform as xform_utils
from pxr import UsdGeom,Gf

USD_PATH=r"C:\Users\sr1ja\Downloads\PnP_test\PnP_test02.usd"
OUTPUT_DIR=r"C:\Users\sr1ja\Downloads\PnP_test\PnP_ground_truth\observation"
FIDUCIAL_PATHS=[f"/World/Fiducials_{i:02d}" for i in range(1,14)]
REFERENCE_CAMERA_PATH="/World/Camera_01"
CAPTURE_CAMERA_PATH="/World/CaptureCamera"
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
RESOLUTION=(IMAGE_WIDTH,IMAGE_HEIGHT)
POSITION_JITTER_RADIUS=0.0
WARMUP_FRAMES=60

def warp_to_numpy(value):
    return np.array(value.numpy(),dtype=np.float64) if hasattr(value,"numpy") else np.array(value,dtype=np.float64)

def get_world_pose_numpy(prim):
    position,orientation=xform_utils.get_world_pose(prim)
    return warp_to_numpy(position),warp_to_numpy(orientation)

def quaternion_wxyz_to_rotation_matrix(q):
    q=np.asarray(q,dtype=np.float64).reshape(4)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)

def get_camera_intrinsics(prim):
    focal_length=float(prim.GetAttribute("focalLength").Get())
    horizontal_aperture=float(prim.GetAttribute("horizontalAperture").Get())
    vertical_aperture=float(prim.GetAttribute("verticalAperture").Get())
    h_offset_attr=prim.GetAttribute("horizontalApertureOffset")
    v_offset_attr=prim.GetAttribute("verticalApertureOffset")
    h_offset=float(h_offset_attr.Get()) if h_offset_attr.IsValid() and h_offset_attr.Get() is not None else 0.0
    v_offset=float(v_offset_attr.Get()) if v_offset_attr.IsValid() and v_offset_attr.Get() is not None else 0.0
    fx=IMAGE_WIDTH*focal_length/horizontal_aperture
    fy=IMAGE_HEIGHT*focal_length/vertical_aperture
    cx=IMAGE_WIDTH/2.0+h_offset*IMAGE_WIDTH/horizontal_aperture
    cy=IMAGE_HEIGHT/2.0+v_offset*IMAGE_HEIGHT/vertical_aperture
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
    projection_attr=prim.GetAttribute("projection")
    projection=str(projection_attr.Get()) if projection_attr.IsValid() else "perspective"
    clipping_attr=prim.GetAttribute("clippingRange")
    clipping_range=None
    if clipping_attr.IsValid() and clipping_attr.Get() is not None:
        value=clipping_attr.Get()
        clipping_range=[float(value[0]),float(value[1])]
    return {"projection":projection,"image_width":IMAGE_WIDTH,"image_height":IMAGE_HEIGHT,"focal_length":focal_length,"horizontal_aperture":horizontal_aperture,"vertical_aperture":vertical_aperture,"horizontal_aperture_offset":h_offset,"vertical_aperture_offset":v_offset,"fx":fx,"fy":fy,"cx":cx,"cy":cy,"K":K.tolist(),"distortion_model":"none","distortion_coefficients":[0.0,0.0,0.0,0.0,0.0],"horizontal_fov_degrees":math.degrees(2*math.atan(horizontal_aperture/(2*focal_length))),"vertical_fov_degrees":math.degrees(2*math.atan(vertical_aperture/(2*focal_length))),"clipping_range":clipping_range}

def get_opencv_extrinsics(camera_position,camera_orientation):
    R_camera_to_world_isaac=quaternion_wxyz_to_rotation_matrix(camera_orientation)
    R_world_to_camera_isaac=R_camera_to_world_isaac.T
    S=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=np.float64)
    R_world_to_camera_opencv=S@R_world_to_camera_isaac
    C=np.asarray(camera_position,dtype=np.float64).reshape(3)
    t=-R_world_to_camera_opencv@C
    T=np.eye(4,dtype=np.float64)
    T[:3,:3]=R_world_to_camera_opencv
    T[:3,3]=t
    camera_center=-R_world_to_camera_opencv.T@t
    return R_world_to_camera_opencv,t,T,camera_center

def project_points(points_world,K,R,t):
    points_world=np.asarray(points_world,dtype=np.float64)
    camera_points=(R@points_world.T).T+t.reshape(1,3)
    projected=[]
    for X,Y,Z in camera_points:
        if Z<=0: projected.append(None)
        else: projected.append([float(K[0,0]*X/Z+K[0,2]),float(K[1,1]*Y/Z+K[1,2])])
    return projected,camera_points

def sample_position_near_reference(reference_position,radius):
    reference_position=np.asarray(reference_position,dtype=np.float64)
    if radius<=0: return reference_position.copy()
    direction=np.random.normal(size=3)
    direction/=np.linalg.norm(direction)
    return reference_position+direction*np.random.uniform(0.0,radius)

def create_capture_camera(stage):
    reference_prim=stage.GetPrimAtPath(REFERENCE_CAMERA_PATH)
    if not reference_prim.IsValid(): raise RuntimeError(f"Reference camera not found: {REFERENCE_CAMERA_PATH}")
    if reference_prim.GetTypeName()!="Camera": raise RuntimeError(f"{REFERENCE_CAMERA_PATH} is not a Camera")
    reference_position,reference_orientation=get_world_pose_numpy(reference_prim)
    print("\n========== REFERENCE CAMERA ==========")
    print("Path:",REFERENCE_CAMERA_PATH)
    print("Position:",reference_position)
    print("Orientation WXYZ:",reference_orientation)
    capture_position=sample_position_near_reference(reference_position,POSITION_JITTER_RADIUS)
    existing=stage.GetPrimAtPath(CAPTURE_CAMERA_PATH)
    if existing.IsValid(): stage.RemovePrim(CAPTURE_CAMERA_PATH)
    capture_prim=stage.DefinePrim(CAPTURE_CAMERA_PATH,"Camera")
    if not capture_prim.IsValid(): raise RuntimeError(f"Failed to create {CAPTURE_CAMERA_PATH}")

    reference_projection_attr=reference_prim.GetAttribute("projection")
    reference_focal_attr=reference_prim.GetAttribute("focalLength")
    reference_hap_attr=reference_prim.GetAttribute("horizontalAperture")
    reference_vap_attr=reference_prim.GetAttribute("verticalAperture")
    capture_camera=UsdGeom.Camera(capture_prim)

    if reference_projection_attr.IsValid() and reference_projection_attr.Get() is not None:
        capture_camera.GetProjectionAttr().Set(reference_projection_attr.Get())
    capture_camera.GetFocalLengthAttr().Set(float(reference_focal_attr.Get()))
    capture_camera.GetHorizontalApertureAttr().Set(float(reference_hap_attr.Get()))
    capture_camera.GetVerticalApertureAttr().Set(float(reference_vap_attr.Get()))

    for attr_name in ["horizontalApertureOffset","verticalApertureOffset"]:
        src=reference_prim.GetAttribute(attr_name)
        if src.IsValid() and src.Get() is not None:
            capture_prim.GetAttribute(attr_name).Set(float(src.Get()))

    src_clip=reference_prim.GetAttribute("clippingRange")
    if src_clip.IsValid() and src_clip.Get() is not None:
        value=src_clip.Get()
        capture_camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(value[0]),float(value[1])))

    xformable=UsdGeom.Xformable(capture_prim)
    translate_op=xformable.AddTranslateOp()
    orient_op=xformable.AddOrientOp()
    translate_op.Set(Gf.Vec3d(float(capture_position[0]),float(capture_position[1]),float(capture_position[2])))

    # IMPORTANT: Isaac Sim created the orient op as GfQuatf.
    # reference_orientation is WXYZ, so construct Gf.Quatf.
    orient_op.Set(Gf.Quatf(float(reference_orientation[0]),Gf.Vec3f(float(reference_orientation[1]),float(reference_orientation[2]),float(reference_orientation[3]))))

    print("\n========== NEW CAPTURE CAMERA ==========")
    print("Path:",CAPTURE_CAMERA_PATH)
    print("Reference position:",reference_position)
    print("Capture position:",capture_position)
    print("Position difference:",np.linalg.norm(capture_position-reference_position))
    print("Using Camera_01 orientation unchanged.")
    return capture_prim,reference_position,reference_orientation,capture_position

writer=None
render_product=None

try:
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    print("="*70)
    print("LOADING USD")
    print("="*70)
    print(USD_PATH)
    omni.usd.get_context().open_stage(USD_PATH)

    while True:
        simulation_app.update()
        stage=omni.usd.get_context().get_stage()
        if stage is not None and all(stage.GetPrimAtPath(p).IsValid() for p in FIDUCIAL_PATHS) and stage.GetPrimAtPath(REFERENCE_CAMERA_PATH).IsValid(): break

    print("\n========== SCENE READY ==========")
    print("\n========== READING FIDUCIALS ==========")
    fiducial_data={}
    for path in FIDUCIAL_PATHS:
        prim=stage.GetPrimAtPath(path)
        position,orientation=get_world_pose_numpy(prim)
        fiducial_data[path]={"position_world":position.tolist(),"orientation_wxyz":orientation.tolist()}
        print(f"{path}: {position}")

    camera_prim,reference_position,reference_orientation,capture_position=create_capture_camera(stage)

    for _ in range(5):
        simulation_app.update()

    actual_position,actual_orientation=get_world_pose_numpy(camera_prim)

    print("\n========== CAPTURE CAMERA POSE ==========")
    print("Position:",actual_position)
    print("Orientation WXYZ:",actual_orientation)
    print("Distance from Camera_01:",np.linalg.norm(actual_position-reference_position),"m")

    intrinsics=get_camera_intrinsics(camera_prim)
    K=np.array(intrinsics["K"],dtype=np.float64)

    print("\n========== CAMERA INTRINSICS ==========")
    print("K:")
    print(K)

    R_wc,t_wc,T_wc,camera_center=get_opencv_extrinsics(actual_position,actual_orientation)

    print("\n========== OPENCV EXTRINSICS ==========")
    print("R:")
    print(R_wc)
    print("t:")
    print(t_wc)
    print("Camera center:")
    print(camera_center)

    print("\n========== GENERATING PNP INPUT ==========")
    fiducial_positions=np.array([fiducial_data[p]["position_world"] for p in FIDUCIAL_PATHS],dtype=np.float64)
    projected_points,camera_points=project_points(fiducial_positions,K,R_wc,t_wc)
    correspondences={}
    visible_count=0

    for i,path in enumerate(FIDUCIAL_PATHS):
        pixel=projected_points[i]
        if pixel is not None and 0<=pixel[0]<IMAGE_WIDTH and 0<=pixel[1]<IMAGE_HEIGHT:
            visible_count+=1
        correspondences[path]={"point_3d_world":fiducial_positions[i].tolist(),"point_3d_camera_opencv":camera_points[i].tolist(),"point_2d_pixel":pixel}
        print(f"{path}: 3D={fiducial_positions[i]} 2D={pixel}")

    print(f"\nIn front + in image: {visible_count}/{len(FIDUCIAL_PATHS)}")

    print("\n========== CREATING RENDER PRODUCT ==========")
    render_product=rep.create.render_product(CAPTURE_CAMERA_PATH,RESOLUTION)
    writer=rep.writers.get("BasicWriter")
    writer.initialize(output_dir=OUTPUT_DIR,rgb=True)
    writer.attach(render_product)
    rep.orchestrator.set_capture_on_play(False)

    print("\n========== RENDERER WARM-UP ==========")
    for i in range(WARMUP_FRAMES):
        simulation_app.update()
        if i%10==0:
            print(f"Renderer warm-up: {i}/{WARMUP_FRAMES}")

    print("\n========== CAPTURING OBSERVATION ==========")
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    print("Capture complete.")

    pngs=glob.glob(os.path.join(OUTPUT_DIR,"**","*.png"),recursive=True)
    if not pngs:
        raise RuntimeError(f"No PNG generated in {OUTPUT_DIR}")

    image_path=os.path.join(OUTPUT_DIR,"observation.png")
    generated_png=max(pngs,key=os.path.getmtime)

    if os.path.abspath(generated_png)!=os.path.abspath(image_path):
        if os.path.exists(image_path):
            os.remove(image_path)
        os.replace(generated_png,image_path)

    print("Image saved:",image_path)

    observation={
        "metadata":{
            "type":"observation",
            "usd_file":USD_PATH,
            "image_resolution":[IMAGE_WIDTH,IMAGE_HEIGHT],
            "reference_camera":REFERENCE_CAMERA_PATH,
            "capture_camera":CAPTURE_CAMERA_PATH,
            "position_jitter_radius":POSITION_JITTER_RADIUS,
            "coordinate_systems":{
                "isaac_world":"Isaac Sim world coordinates",
                "isaac_camera":"+X right, +Y up, -Z forward",
                "opencv_camera":"+X right, +Y down, +Z forward"
            },
            "intrinsic_model":"Perspective pinhole",
            "distortion":"Zero distortion"
        },
        "fiducials":fiducial_data,
        "camera":{
            "path":CAPTURE_CAMERA_PATH,
            "reference_camera":{
                "path":REFERENCE_CAMERA_PATH,
                "position_world":reference_position.tolist(),
                "orientation_wxyz":reference_orientation.tolist()
            },
            "intrinsics":intrinsics,
            "ground_truth_pose":{
                "position_world":actual_position.tolist(),
                "orientation_wxyz":actual_orientation.tolist()
            },
            "opencv_extrinsics":{
                "R_world_to_camera":R_wc.tolist(),
                "t_world_to_camera":t_wc.tolist(),
                "T_world_to_camera":T_wc.tolist(),
                "camera_center_world":camera_center.tolist()
            },
            "correspondences":correspondences,
            "image":"observation.png"
        }
    }

    json_path=os.path.join(OUTPUT_DIR,"observation.json")
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(observation,f,indent=4,allow_nan=False)

    print("\n"+"="*70)
    print("OBSERVATION COMPLETE")
    print("="*70)
    print("Reference camera:",REFERENCE_CAMERA_PATH)
    print("Capture camera:",CAPTURE_CAMERA_PATH)
    print("Reference position:",reference_position)
    print("Capture position:",actual_position)
    print("Position delta:",np.linalg.norm(actual_position-reference_position),"m")
    print("Image:",image_path)
    print("JSON:",json_path)
    print("PnP correspondences:",len(correspondences))
    print("="*70)

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