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
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T
import yaml
from moviepy.editor import ImageSequenceClip
from tqdm import trange

from diffusion_policy.common.camera_utils import CameraMover
from diffusion_policy.common.multi_camera_utils import CameraMoverMultiWrist, get_observation_from_added_cams, get_observation_with_added_cams

MOVE_CAM = False
FOV=90
# HEIGHT = 1080
# WIDTH = 1920
HEIGHT = 800
WIDTH = 800
gopro = 'gopro' # 'orbslam'
crop_output = True # 'False
class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'
    
def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: DistortMode = DistortMode.LINEAR, crop_output: bool = False,
                  crop_type: str = "middle") -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): RGB image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.
        crop_type (str): How to crop.
            "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
            "middle": We take the widest points along the middle of the image (height and width). There will be black
                      pixels on the corners. To counter this, original image has to be higher FOV than the desired output.

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    else:
        raise RuntimeError('Image shape of {} is not supported. Valid shape should be (H,W,3)'.format(imshape))

    imdtype = img.dtype

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    if chan == 1:
        img = np.expand_dims(img, 2)
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                               bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
        # Crop rectangle from resulting distorted image
        # Get mapping from undistorted to distorted
        distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
        cam_intr_inv = np.linalg.inv(cam_intr)
        distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
        distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        distorted_px = distorted_px.reshape((h, w, 2))
        if crop_type == "corner":
            # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
            top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int)
            bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int)
            img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist

def animate(imgs, filename='animation.mp4', _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=False)
import random

def split_into_sets(n, a, b, c):
    if a + b + c != n:
        raise ValueError("The sum of a, b, and c must be equal to n")

    numbers = list(range(n))
    random.shuffle(numbers)

    set_a = numbers[:a]
    set_b = numbers[a:a+b]
    set_c = numbers[a+b:]

    return set_a, set_b, set_c 
def animate_tricam(imgs_dict, keys, filename='animation.mp4', _return=True, fps=10):
    left = keys[0]
    middle = keys[1]
    right = keys[2]
    imgs_list = []
    for time in range(len(imgs_dict[left])):
        concat_img_list = []
        for key in keys:
            concat_img_list.append(imgs_dict[key][time])
        concat_img = np.concatenate(concat_img_list, axis=1)
        imgs_list.append(concat_img)
    imgs = ImageSequenceClip(imgs_list, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=False)

import math
gopro_dict={
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
        }
    }
f = 0.5 * HEIGHT / math.tan(FOV * math.pi / 360)
cam_intrinsic_matrix = np.array([[f, 0, WIDTH/2], 
                                 [0, f, HEIGHT/2], 
                                 [0, 0, 1]])

D = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182], dtype=np.float32)
D_gopro_umi = np.array([gopro_dict["intrinsics"]["radial_distortion_1"],\
            gopro_dict["intrinsics"]["radial_distortion_2"],\
            gopro_dict["intrinsics"]["radial_distortion_3"],\
            gopro_dict["intrinsics"]["radial_distortion_4"]], dtype=np.float32)
if gopro == 'orbslam': # orbslam fisheye D info
    D = D
elif gopro == 'gopro':
    D = D_gopro_umi
    

for wrist_delta_angle in [45]:
    object = 'can'
    object_dir = f'{object}_panorama_move_fov_{FOV}'
    dataset_path = 'data/robomimic/datasets/{}/ph/image.hdf5'.format(object)
    if object in ['transport']:
        camera_name = 'shouldercamera0'
    else:
        camera_name = 'agentview'
    camera_name1 = 'frontview'
    camera_name2 = 'birdview'
    camera_name3 = 'robot0_eye_in_hand'
    # camera_name4 = 'robot0_eye_in_hand'
    # camera_name5 = 'robot0_eye_in_hand'
    
    f = h5py.File(dataset_path, 'r')
    states = f['data']['demo_0']['states']
    actions = f['data']['demo_0']['actions']
    images = [x for x in f['data']['demo_0']['obs']['{}_image'.format(camera_name)]]
    animate(images, filename = '{}.mp4'.format(camera_name))
 
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
    pre_camera_mover_xml = env.env.sim.model.get_xml()
    with open(os.path.join(os.getcwd(), 'pre_camera_mover_xml.xml'), 'w') as f:
        f.write(pre_camera_mover_xml)
        
    # added_camera_list = [camera_name4, camera_name5]
    # wrist_camera_list = [camera_name3, camera_name4, camera_name5]
    wrist_camera_list = [camera_name]
    camera_name = camera_name


    camera_mover = CameraMover(
        env=env.env,
        camera=camera_name,
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
    train_split = traj_len-test_split-val_split
  
    print(f"traj_len: {traj_len},train: {train_split}, val: {val_split}, test: {test_split}")
    # val_split = (traj_len-train_split)//2
    # test_split = traj_len-train_split-val_split
    # train_idx, val_idx, test_idx = split_into_sets(traj_len, a=train_split, b=val_split, c=test_split)
  
    img_dict={}
    for camera_name_obs in wrist_camera_list:
        img_dict[camera_name_obs] = list()
    
    json_dict_train = {}
    json_dict_train['camera_angle_x'] = FOV/180*np.pi
    json_dict_frames = []
    stage = 'train'
    moving_animation_path = os.path.join(f'/local/real/chuerpan/repo/4DGaussians/data/robomimic/{object_dir}/{stage}')
    if not os.path.exists(moving_animation_path):
        os.makedirs(moving_animation_path)
    counter_train = 0
    for t in trange(train_split):


        action = actions[t].copy()

        v_pos = action[:3]

        obs = env.reset_to({"states" : states[t+1]})
 
        # observation, reward, done, info = env.step(np.zeros(actions[t].shape))
        # env.env.sim.set_state_from_flattened(states[t+1])
        # env.env.sim.forward()
        new_state = env.env.sim.get_state()

        for camera_name_obs in wrist_camera_list:
            json_dict_frame_i = {}
            file_path = f'./{stage}/r_{str(counter_train).zfill(3)}'
            json_dict_frame_i['file_path'] = file_path
            
            img = obs[f'{camera_name_obs}_image']
            img = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8) # ()(H, W, 3)
            image = Image.fromarray(img)
            image.save(os.path.join(moving_animation_path, f'{file_path.split("/")[-1]}.png'))
        
            img_dict[camera_name_obs].append(img) 
            cur_camera_pos, cur_camera_quat = camera_mover.get_camera_pose()
            camera_rot = T.quat2mat(cur_camera_quat)
            
 
            cam_T = np.eye(4)
            cam_T[:3,:3] = np.copy(camera_rot)
            cam_T[:3,3] = np.copy(cur_camera_pos)
            json_dict_frame_i['transform_matrix'] = cam_T.tolist()
            json_dict_frame_i['time'] = float(new_state.time)
            _, quat = T.mat2pose(cam_T)
            json_dict_frame_i['rotation'] = float(np.linalg.norm(T.quat2axisangle(np.copy(quat))))
            
            dot_product= list()
            angle = 3 # degree
            for sgn in [1]:
          
                rad = sgn * np.pi * angle / 180.0
                R = T.rotation_matrix(rad, [0, 0, 1], point=None)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = camera_rot
                camera_pose[:3, 3] = cur_camera_pos
                camera_pose = R @ camera_pose
                dot_product.append(camera_pose[:3, 3].dot(v_pos))
            max_idx = np.argmin(np.abs(dot_product))
            if max_idx == 0:
                camera_mover.rotate_camera_world(None, [0, 0, 1], angle)
            elif max_idx == 2:
                camera_mover.rotate_camera_world(None, [0, 0, 1], -angle)
 
            counter_train+=1
            json_dict_frames.append(json_dict_frame_i)
            

    json_dict_train['frames'] = json_dict_frames
    with open(os.path.join(moving_animation_path.split(stage)[0], f'transforms_{stage}.json'), 'w') as file:
        json.dump(json_dict_train, file)

    json_dict_val = {}
    json_dict_val['camera_angle_x'] = FOV/180*np.pi
    json_dict_frames_val = []
    stage = 'val'
    moving_animation_path = os.path.join(f'/local/real/chuerpan/repo/4DGaussians/data/robomimic/{object_dir}/{stage}')
    if not os.path.exists(moving_animation_path):
        os.makedirs(moving_animation_path)
    
    counter_val = 0
    for t in trange(train_split, train_split+val_split):

        action = actions[t].copy()
 
        v_pos = action[:3]
 
        obs = env.reset_to({"states" : states[t+1]})
        new_state = env.env.sim.get_state()
 
        for camera_name_obs in wrist_camera_list:
            json_dict_frame_i = {}
            file_path = f'./{stage}/r_{str(counter_val).zfill(3)}'
            json_dict_frame_i['file_path'] = file_path
            img = obs[f'{camera_name_obs}_image']
            img = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8) # ()(H, W, 3)
            image = Image.fromarray(img)
            image.save(os.path.join(moving_animation_path, f'{file_path.split("/")[-1]}.png'))

            img_dict[camera_name_obs].append(img) 
            cur_camera_pos, cur_camera_quat = camera_mover.get_camera_pose()
            camera_rot = T.quat2mat(cur_camera_quat)
            
 
            cam_T = np.eye(4)
            cam_T[:3,:3] = np.copy(camera_rot)
            cam_T[:3,3] = np.copy(cur_camera_pos)
            json_dict_frame_i['transform_matrix'] = cam_T.tolist()
            json_dict_frame_i['time'] = float(new_state.time)
            _, quat = T.mat2pose(cam_T)
            json_dict_frame_i['rotation'] = float(np.linalg.norm(T.quat2axisangle(np.copy(quat))))
            
            dot_product= list()
            angle = 3 # degree
            for sgn in [1]:
       
                rad = sgn * np.pi * angle / 180.0
                R = T.rotation_matrix(rad, [0, 0, 1], point=None)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = camera_rot
                camera_pose[:3, 3] = cur_camera_pos
                camera_pose = R @ camera_pose
                dot_product.append(camera_pose[:3, 3].dot(v_pos))
            max_idx = np.argmin(np.abs(dot_product))
            if max_idx == 0:
                camera_mover.rotate_camera_world(None, [0, 0, 1], angle)
            elif max_idx == 2:
                camera_mover.rotate_camera_world(None, [0, 0, 1], -angle)
 
            counter_val+=1
            json_dict_frames_val.append(json_dict_frame_i)
 
    json_dict_val['frames'] = json_dict_frames_val
    with open(os.path.join(moving_animation_path.split(stage)[0], f'transforms_{stage}.json'), 'w') as file:
        json.dump(json_dict_val, file)

    json_dict_test = {}
    json_dict_test['camera_angle_x'] = FOV/180*np.pi
    json_dict_frames_test = []
    stage = 'test'
    moving_animation_path = os.path.join(f'/local/real/chuerpan/repo/4DGaussians/data/robomimic/{object_dir}/{stage}')
    if not os.path.exists(moving_animation_path):
        os.makedirs(moving_animation_path)
    counter_test = 0
    for t in trange(train_split+val_split, traj_len):
 
        action = actions[t].copy()
 
        v_pos = action[:3]
 
        obs = env.reset_to({"states" : states[t+1]})
        new_state = env.env.sim.get_state()
 
        for camera_name_obs in wrist_camera_list:
            json_dict_frame_i = {}
            file_path = f'./{stage}/r_{str(counter_test).zfill(3)}'
            json_dict_frame_i['file_path'] = file_path
            
            img = obs[f'{camera_name_obs}_image']
            img = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8) # ()(H, W, 3)
            image = Image.fromarray(img)
            image.save(os.path.join(moving_animation_path, f'{file_path.split("/")[-1]}.png'))

            img_dict[camera_name_obs].append(img) 
            cur_camera_pos, cur_camera_quat = camera_mover.get_camera_pose()
            camera_rot = T.quat2mat(cur_camera_quat)
            
 
            cam_T = np.eye(4)
            cam_T[:3,:3] = np.copy(camera_rot)
            cam_T[:3,3] = np.copy(cur_camera_pos)
            json_dict_frame_i['transform_matrix'] = cam_T.tolist()
            json_dict_frame_i['time'] = float(new_state.time)
            _, quat = T.mat2pose(cam_T)
            json_dict_frame_i['rotation'] = float(np.linalg.norm(T.quat2axisangle(np.copy(quat))))
            
            dot_product= list()
            angle = 3 # degree
            for sgn in [1]:
                rad = sgn * np.pi * angle / 180.0
                R = T.rotation_matrix(rad, [0, 0, 1], point=None)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = camera_rot
                camera_pose[:3, 3] = cur_camera_pos
                camera_pose = R @ camera_pose
                dot_product.append(camera_pose[:3, 3].dot(v_pos))
            max_idx = np.argmin(np.abs(dot_product))
            if max_idx == 0:
                camera_mover.rotate_camera_world(None, [0, 0, 1], angle)
            elif max_idx == 2:
                camera_mover.rotate_camera_world(None, [0, 0, 1], -angle)
 
            counter_test+=1
            json_dict_frames_test.append(json_dict_frame_i)
 
    json_dict_test['frames'] = json_dict_frames_test
    with open(os.path.join(moving_animation_path.split(stage)[0], f'transforms_{stage}.json'), 'w') as file:
        json.dump(json_dict_test, file)

 
    # animate_tricam(imgs_dict=img_dict, keys=[camera_name4, camera_name3, camera_name5],\
    #     filename=os.path.join(moving_animation_path, f'{object}_tricam_moving_animation_{wrist_delta_angle}_fov_{FOV}_{gopro}_cropOP_{crop_output}.mp4'))
 
# %%
