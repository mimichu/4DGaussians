import json
import numpy as np
import os

def merge_data(path1, path2, new_dir_name):
    """
    Args:
        path1 (str), data path for one data source, in the form path1/{val, train, test, *.json}
        path2 (str), data path for one data source, in the form path1/{val, train, test, *.json}
        new_dir_name (str), data dir name to store the newly merge data
    """
    
    # merge train imagees
    os.load(path1)
    
    
import os
import shutil

def list_images(directory):
    """List all image files in a directory."""
    return [f for f in os.listdir(directory) if f.endswith('.png')]

def copy_and_reindex_images(src_dir_a, src_dir_b, dest_dir):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # List images in A/train and B/train
    images_a = list_images(src_dir_a)
    images_b = list_images(src_dir_b)

    # Copy images from A/train to C/train
    for image in images_a:
        shutil.copy(os.path.join(src_dir_a, image), os.path.join(dest_dir, image))

    # Calculate starting index for images from B/train
    start_index = len(images_a)

    # Re-index and copy images from B/train to C/train
    for index, image in enumerate(images_b, start=start_index):
        new_image_name = f"r_{str(index).zfill(3)}.png"
        shutil.copy(os.path.join(src_dir_b, image), os.path.join(dest_dir, new_image_name))

# Usage
src_dir_a = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_panorama_move_fov_90_static_wrist/train'  # Replace with actual path to A/train
src_dir_b = '/local/real/chuerpan/repo/4DGaussians/data/robomimic_old/can_panorama_move_fov_90/train'   # Replace with actual path to B/train
dest_dir = '/local/real/chuerpan/repo/4DGaussians/data/robomimic/can_fov_90_static_train_pano_wrist_test_wrist/train'   # Replace with actual path to C/train
copy_and_reindex_images(src_dir_a, src_dir_b, dest_dir)
