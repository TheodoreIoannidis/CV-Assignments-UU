import glm
import random
import numpy as np
from task_3 import *


block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def build_projection_lut(cams, voxel_space):
    num_voxels = voxel_space.shape[0]
    lut = np.empty((len(cams), num_voxels, 2), dtype=int)

    # Project the entire voxel space once per camera
    for c_idx, cam in enumerate(cams):
        camera_matrix, dist_coeffs, rvec, tvec = load_config(cam)
        image_points, _ = cv2.projectPoints(voxel_space.astype(np.float64),
                                            rvec, tvec, camera_matrix, dist_coeffs)
        # Round to integer pixel coords
        points_rounded = np.rint(image_points[:, 0, :]).astype(int)
        lut[c_idx] = points_rounded

    return lut


# This part is to create the reconstruction via the lookup table - more efficient
def reconstruct_voxels(cams, voxel_space, lut, frame_masks, cam_votes=2):
    num_voxels = voxel_space.shape[0]
    votes = np.zeros(num_voxels, dtype=int)

    for c_idx, cam in enumerate(cams):
        mask = frame_masks[cam]   
        px = lut[c_idx, :, 0]
        py = lut[c_idx, :, 1]
        valid = (
            (px >= 0) & (px < mask.shape[1]) &
            (py >= 0) & (py < mask.shape[0])
        )

        # For valid coords, check if mask == 255
        vote_contrib = np.zeros(num_voxels, dtype=int)
        vote_contrib[valid] = (mask[py[valid], px[valid]] == 255).astype(int)
        votes += vote_contrib

    voxel_on = votes >= cam_votes
    return voxel_on






def set_voxel_positions(width, height, depth):

    x_range = (-width / 2 * block_size, width / 2 * block_size)
    y_range = (0, height * block_size)
    z_range = (-depth / 2 * block_size, depth / 2 * block_size)

    x_vals = np.arange(x_range[0], x_range[1], block_size)
    y_vals = np.arange(y_range[0], y_range[1], block_size)
    z_vals = np.arange(z_range[0], z_range[1], block_size)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    voxel_space = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    
    cams = [f"cam{i}" for i in range(1, 5)]
    lut = build_projection_lut(cams, voxel_space)

    frame_masks = {}
    for cam in cams:
    
        mask = cv2.imread(f"data/{cam}/foreground_mask.png", cv2.IMREAD_GRAYSCALE)
        frame_masks[cam] = mask
        
    voxel_mask = reconstruct_voxels(cams, voxel_space, lut, frame_masks, cam_votes=4) #change how many votes needed - when 4, it means that if a point is not seen by one camera it is auto off 
    on_voxels = voxel_space[voxel_mask]
    num_on_voxels = np.count_nonzero(voxel_mask)
    print("Number of voxels turned on:", num_on_voxels)
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range
    colors = []
    for v in on_voxels:
        norm_x = (v[0] - min_x) / (max_x - min_x)
        norm_y = (v[1] - min_y) / (max_y - min_y)
        norm_z = (v[2] - min_z) / (max_z - min_z)
        colors.append([norm_x, norm_y, norm_z])

    data = on_voxels.tolist()
    return data, colors

'''
def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.

    #set boundaries for grid
    x_range = (-width / 2 * block_size, width / 2 * block_size)
    y_range = (0, height * block_size)
    z_range = (-depth / 2 * block_size, depth / 2 * block_size)

    #create grid 
    x_vals = np.arange(x_range[0], x_range[1], block_size)
    y_vals = np.arange(y_range[0], y_range[1], block_size)
    z_vals = np.arange(z_range[0], z_range[1], block_size)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    voxel_space = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    cams = [f"cam{i}" for i in range(1, 5)]
    lut = build_projection_lut(cams, voxel_space)
    frame_masks = {cam: cv2.imread(f"data/{cam}/foreground_mask.png", cv2.IMREAD_GRAYSCALE) for cam in cams}

    voxel_mask = reconstruct_voxels(cams, voxel_space, lut, frame_masks, cam_votes=2)
    on_voxels = voxel_space[voxel_mask]
    
    min_x, max_x = x_range
    min_y, max_y = y_range
    min_z, max_z = z_range
    colors = []

    #colors based on normalized positions
    for v in on_voxels:
        norm_x = (v[0] - min_x) / (max_x - min_x)
        norm_y = (v[1] - min_y) / (max_y - min_y)
        norm_z = (v[2] - min_z) / (max_z - min_z)
        colors.append([norm_x, norm_y, norm_z])
    
    
    data = on_voxels.tolist()
    return data, colors
'''

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # the camera center is computed as: C = -R^T * tvec,

    cams = [f"cam{i}" for i in range(1, 5)]
    positions = []

    for cam in cams:
            camera_matrix, dist_coeffs, rvec, tvec = load_config(cam)

            #camera center in world coordinates. 
            cam_center = -np.matmul(rvec.T, tvec).flatten()
            print(cam_center)
            positions.append(cam_center.tolist())

    print("centers", positions)
    colors = [[1.0, 0, 0],   # Redχ
            [0, 1.0, 0],   # Green
            [0, 0, 1.0],   # Blue
            [1.0, 1.0, 0]] # Yellow
    
    return positions, colors



def numpy_to_glm_mat4(R):
    # 3x3 np rotation matrix to 4x4 glm matrix.
    M = glm.mat4(1.0)  # identity matrix
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R[i, j])
    return M



def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

    cams = [f"cam{i}" for i in range(1, 5)]
    rotation_matrices = []
    for cam in cams: 
        camera_matrix, dist_coeffs, rvec, tvec = load_config(cam) 
        #camera orientation in world coordinates  
        R_world = rvec.T
        M = numpy_to_glm_mat4(R_world)
        rotation_matrices.append(M)
    
    return rotation_matrices

