#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_w2c_transform_fix(frame):
    c2w = np.array(frame["transform_matrix"])
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    return w2c

def get_c2w_transform(frame):
    c2w = np.array(frame["transform_matrix"])
    return c2w

def plot_camera_pose(ax, pose, size=0.1):
    """ Plot a single camera pose """
    # Extract translation and rotation
    t = pose[:3, 3:4]
    R = pose[:3, :3]

    # Create camera coordinate frame
    camera_shape = np.array(
        [[-size, size, 1],
         [size, size, 1],
         [size, -size, 1],
         [-size ,-size, 1],
         [0,0,0]]
    ).T # 3,5
 
    camera_shape_transformed = R@camera_shape+t
    # cartesian_axis = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]]) # 3,3
    # a, b, c, d, e  = R @ camera_shape +# 3,5
    
    # ax.quiver(*t, a[0], a[1], a[2], color='r')
    # ax.quiver(*t, b[0], b[1], b[2], color='g')
    # ax.quiver(*t, c[0], c[1], c[2], color='b')
 
    polygons = [camera_shape_transformed.T]
    poly3d = Poly3DCollection(polygons, alpha=0.5)
    ax.add_collection3d(poly3d)

def get_camera_polygons(camera_poses, size = 0.1):
    """
    Args:
        camera_poses, [(4,4)], list of SE(3) matrices
    Return:
        polygons, [(5,3)], list of np.array (5,3) for cameras
    """
    polygons = []
    for pose in camera_poses:
        t = pose[:3, 3:4]
        R = pose[:3, :3]

        # Create camera coordinate frame
        camera_shape = np.array(
            [[-size, size, 1],
            [size, size, 1],
            [size, -size, 1],
            [-size ,-size, 1],
            [0,0,0]]
        ).T # 3,5
    
        camera_shape_transformed = R@camera_shape+t
        polygons.append(camera_shape_transformed.T)
    return polygons
    
    
if __name__ == '__main__':
    json_file_path = '/local/real/chuerpan/repo/4DGaussians/data/robomimic_old/can_panorama_move_fov_45_static_200_01/transforms_train.json'

    with open(json_file_path) as json_file:
        contents = json.load(json_file)
  
    frames = contents["frames"]
    camera_poses = []
    for frame in frames:
        c2w_T = get_c2w_transform(frame)
        camera_poses.append(c2w_T)
    test_T = np.eye(4)
    # test_T[:3, 3] = 1
    camera_poses = [test_T]
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    polygons = get_camera_polygons(camera_poses=camera_poses, size = 0.1)
    poly3d = Poly3DCollection(polygons, alpha=0.5)
    ax.add_collection3d(poly3d)
    polygons_arr = np.stack(polygons) # (N, 5, 3)
    polygons_arr = polygons_arr.reshape(polygons_arr.shape[0]*polygons_arr.shape[1], -1) # (N*5, 3)
    ax.set_xlim([min(polygons_arr[:,0]), max(polygons_arr[:,0])])
    ax.set_ylim([min(polygons_arr[:,1]), max(polygons_arr[:,1])])
    ax.set_zlim([min(polygons_arr[:,2]), max(polygons_arr[:,2])])
    

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    import plotly.tools as tls
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig.write_html('plot_cam.html')

# %%
