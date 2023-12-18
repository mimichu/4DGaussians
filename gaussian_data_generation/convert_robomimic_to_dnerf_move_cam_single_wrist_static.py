### Trying to add new fixed cameras to the robot0_eye_in_hand's parent
### Where I am stuck:
### 1) get env_meta from the dataset path
### 2) recreate the env_meta with the then modify the xml to add new cameras
### 2) create env_meta

#%%
import os
import tqdm
import json
import sys
import matplotlib.pyplot as plt
import cv2
import enum
import scipy.interpolate
from PIL import Image
class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'
    
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import collections
import h5py
import matplotlib.pyplot as plt
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import yaml
from diffusion_policy.common.multi_camera_utils import ChangeCameraFOV
from diffusion_policy.gaussians_utils.panorama_cam import create_data_stage_wise_static_wrist

TIME_INTERVAL = 0.1
FOV = 120
TRAJ_LEN = 200
HEIGHT = 800
WIDTH = 800
SEQ_WINDOW = 39
from diffusion_policy.gaussians_utils.animate_utils import animate
import numpy as np
for wrist_delta_angle in [45]:
    object = 'can'
    object_dir = f'{object}_panorama_move_fov_{FOV}_static_wrist'
    dataset_path = 'data/robomimic/datasets/{}/ph/image.hdf5'.format(object)
    if object in ['transport']:
        camera_name = 'shouldercamera0'
    else:
        camera_name = 'agentview'
    camera_name1 = 'frontview'
    camera_name2 = 'birdview'
    camera_name3 = 'robot0_eye_in_hand'
    
    f = h5py.File(dataset_path, 'r')
    states = f['data']['demo_0']['states']
    actions = f['data']['demo_0']['actions']
    images = [x for x in f['data']['demo_0']['obs']['{}_image'.format(camera_name)]]
    
    states = np.concatenate([np.copy(states[:SEQ_WINDOW]), np.copy(states[:SEQ_WINDOW][::-1])])
    actions = np.concatenate([np.copy(actions[:SEQ_WINDOW]), np.copy(actions[:SEQ_WINDOW][::-1])])
    images = images[:SEQ_WINDOW]+ images[:SEQ_WINDOW][::-1]
    
    animate(images, filename = '{}.mp4'.format(camera_name))
#%%
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
    
    wrist_camera_list = [camera_name3]
    camera_name = camera_name3


    camera_mover = ChangeCameraFOV(
        env=env.env,
        camera_name =camera_name,
        fov=FOV
    )
    post_camera_mover_xml = env.env.sim.model.get_xml()

    with open(os.path.join(os.getcwd(), 'post_camera_mover_xml.xml'), 'w') as f:
        f.write(post_camera_mover_xml)

    
    env.reset_to({"states" : states[0]})
    
    camera_name_obs = camera_name3
    traj_len = states.shape[0]
    traj_len = traj_len-1
    val_split = 6
    test_split = 6
    train_split = 30
  
    print(f"traj_len: {traj_len},train: {train_split}, val: {val_split}, test: {test_split}")
    local_save_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic'
    
    train_list = np.arange(train_split).tolist() + np.arange(train_split+12, traj_len,1).tolist()
    val_list = np.arange(train_split, train_split+6, 1).tolist()  
    test_list = np.arange(train_split+6, train_split+12, 1).tolist()  
    
    create_data_stage_wise_static_wrist(stage = 'train', 
                           save_path = local_save_path,
                           env = env,
                           fov = FOV, 
                           object_dir = object_dir,
                           actions = actions,
                           states = states, 
                           wrist_camera_list = wrist_camera_list,
                           time_interval = TIME_INTERVAL, 
                           t_list = train_list)
    
    create_data_stage_wise_static_wrist(stage = 'val', 
                           save_path = local_save_path,
                           env = env,
                           fov = FOV, 
                           object_dir = object_dir,
                           actions = actions,
                           states = states, 
                           wrist_camera_list = wrist_camera_list,
                           time_interval = TIME_INTERVAL, 
                           t_list = val_list)
    
    create_data_stage_wise_static_wrist(stage = 'test', 
                           save_path =local_save_path,
                           env = env,
                           fov = FOV, 
                           object_dir = object_dir,
                           actions = actions,
                           states = states, 
                           wrist_camera_list = wrist_camera_list,
                           time_interval = TIME_INTERVAL, 
                           t_list = test_list)
    
    