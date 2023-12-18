#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import xml.etree.ElementTree as ET
import math

# Utils functions
def get_rgbd(env, camera_name, width, height):
    
    rgb, depth = env.env.sim.render(camera_name = camera_name, width = width, height = height, depth = True)
    depth = np.expand_dims(np.copy(depth), axis=-1) # H, W, 1
    rgbd = np.concatenate([rgb, depth], axis=-1) # H, W, 4 [R, G, B, D]
    return rgbd

def get_fov(env, camera_name):
    xml = env.env.sim.model.get_xml()
    tree = ET.fromstring(xml)
    wb = tree.find("worldbody")
    camera_elem = None
    cameras = wb.findall(".//camera") # find all cameras under worldbody recursively
    parent_map = dict((c.get("name"), p) for p in tree.iter() for c in p) # build a map of {child.name -> parent element}
 
    for camera in cameras:
        if camera.get("name") == camera_name:
            camera_elem = camera
            break
    fovy = camera_elem.get('fovy')
    if fovy == None:
        return 45
    return fovy

#%%
import os
import sys
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
import collections
 
HEIGHT = 800
WIDTH = 800
object = 'can'
dataset_path = '/local/real/chuerpan/repo/zhenjia_dp/diffusion_policy/data/robomimic/datasets/{}/ph/image.hdf5'.format(object)
camera_name = 'agentview'
camera_name1 = 'frontview'
camera_name2 = 'birdview'
camera_name3 = 'robot0_eye_in_hand'
f = h5py.File(dataset_path, 'r')
states = f['data']['demo_0']['states']
actions = f['data']['demo_0']['actions']
obs = f['data']['demo_0']['obs']
images = [x for x in f['data']['demo_0']['obs']['{}_image'.format(camera_name)]]


shape_meta_yaml = """
obs:
    {}_image:
        shape: [3, 800, 800]
        type: rgb
    {}_image:
        shape: [3, 800, 800]
        type: rgb
    {}_image:
        shape: [3, 800, 800]
        type: rgb
    {}_image:
        shape: [3, 800, 800]
        type: rgb
action: 
    shape: [10]
""".format(camera_name, camera_name1, camera_name2, camera_name3)
shape_meta = yaml.load(shape_meta_yaml, Loader=yaml.Loader)
modality_mapping = collections.defaultdict(list)
for key, attr in shape_meta['obs'].items():
    modality_mapping[attr.get('type', 'low_dim')].append(key)
ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
env_meta['env_kwargs']['camera_depths'] = True
env_meta['env_kwargs']['camera_names'].append(camera_name1)
env_meta['env_kwargs']['camera_names'].append(camera_name2)
env_meta['env_kwargs']['camera_names'].append(camera_name3)
env_meta['env_kwargs']['camera_heights'] = HEIGHT
env_meta['env_kwargs']['camera_widths'] = WIDTH

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False, 
    render_offscreen=True,
    use_image_obs=True, 
)

t = 0
action = actions[t].copy()
v_pos = action[:3]
obs = env.reset_to({"states" : states[t+1]})

sim_rgbd = get_rgbd(env=env, camera_name=camera_name, width=WIDTH, height=HEIGHT)
fov = get_fov(env=env, camera_name='agentview')
f = 0.5 * HEIGHT / math.tan(fov * math.pi / 360)
cam_intrinsic_matrix = np.array([[f, 0, WIDTH/2], 
                                 [0, f, HEIGHT/2], 
                                 [0, 0, 1]])
cam_intrinsic_matrix_no_fov_set = np.array([[0, 0, 0], 
                                 [0, 0, 0], 
                                 [0, 0, 1]])


# /local/real/chuerpan/repo/zhenjia_dp/diffusion_policy/data/robomimic/datasets/can/ph/image.hdf5
# python dataset_states_to_obs.py --dataset /local/real/chuerpan/repo/zhenjia_dp/diffusion_policy/data/robomimic/datasets/can/ph/image.hdf5 --output_name depth.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 800 --camera_width 800 --depth


#%%
import open3d as o3d

# depth_o3d = o3d.geometry.Image(sim_rgbd[:,:,3:4].astype('uint8'))
# cam_intrinsic = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(cam_intrinsic_matrix_no_fov_set)
# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(sim_rgbd[:,:,:3].astype('uint8')), depth_o3d, convert_rgb_to_intensity = False)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsic)
K = cam_intrinsic_matrix_no_fov_set
rgb = sim_rgbd[:,:,:3]
depth = sim_rgbd[:,:,3:]
# Create an Open3D RGB-D image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image((rgb * 255).astype(np.uint8)),
    o3d.geometry.Image(depth.astype(np.float32)),
    # depth_scale=1000.0,  # Adjust according to your depth scale
    # depth_trunc=3.0,     # Adjust truncation for better visualization
    convert_rgb_to_intensity=False
)

#%%
# Create point cloud
intrinsic = o3d.camera.PinholeCameraIntrinsic(HEIGHT, WIDTH, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Visualize
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("robomimic_can_points_default.ply", pcd)
 
# # Set up the off-screen renderer with desired width and height
# vis = o3d.visualization.rendering.OffscreenRenderer(640, 480)
# vis.setup_camera(60, pcd.get_center(), pcd.get_center() + [0, 0, 3], [0, -1, 0])
# vis.scene.add_geometry("point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())

# # Capture the image
# image = vis.render_to_image()
# %%
import open3d as o3d
import matplotlib.pyplot as plt
dataset = o3d.data.EaglePointCloud()
pcd_eagle = o3d.io.read_point_cloud('/local/real/chuerpan/repo/4DGaussians/data/dnerf/bouncingballs/points3d.ply')
vis = o3d.visualization.rendering.OffscreenRenderer(640, 480)
vis.setup_camera(60, pcd_eagle.get_center(), pcd_eagle.get_center() + [0, 0, 1], [0, -1, 0])
vis.scene.add_geometry("point_cloud", pcd_eagle, o3d.visualization.rendering.MaterialRecord())

# Capture the image
image = vis.render_to_image()
# %%


