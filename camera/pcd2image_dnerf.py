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
 

 
# %%
import open3d as o3d
import matplotlib.pyplot as plt
dataset = o3d.data.EaglePointCloud()
pcd_eagle = o3d.io.read_point_cloud(dataset.path)
vis = o3d.visualization.rendering.OffscreenRenderer(640, 480)
vis.setup_camera(60, pcd_eagle.get_center(), pcd_eagle.get_center() + [0, 0, 1], [0, -1, 0])
vis.scene.add_geometry("point_cloud", pcd_eagle, o3d.visualization.rendering.MaterialRecord())

# Capture the image
image = vis.render_to_image()
# %%
