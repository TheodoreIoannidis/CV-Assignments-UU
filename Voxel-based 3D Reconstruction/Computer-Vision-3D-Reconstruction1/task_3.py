import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_config(cam): 
    fs = cv2.FileStorage(f"data/{cam}/config.xml", cv2.FILE_STORAGE_READ)    
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()
    return camera_matrix, dist_coeffs, rvec, tvec

def create_voxel_space(x_range, y_range, z_range, resolution=4):
    #adjust resolution (2 will be finer)
    # this is the "region of interest" to check whether 
    x_vals = np.arange(x_range[0], x_range[1], resolution)
    y_vals = np.arange(y_range[0], y_range[1], resolution)
    z_vals = np.arange(z_range[0], z_range[1], resolution)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    voxel_space = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return voxel_space




def reconstruct_voxels(cams, voxel_space, cam_votes=3):
    # project each voxel into each camera using tje calibration
    num_voxels = voxel_space.shape[0]
    votes = np.zeros(num_voxels, dtype=int)

    for cam in cams:
        #load calibration and foreground of camera
        camera_matrix, dist_coeffs, rvec, tvec = load_config(cam) 
        mask = cv2.imread(f"data/{cam}/foreground_mask.png", cv2.IMREAD_GRAYSCALE)

        #project voxels
        image_points, _ = cv2.projectPoints(voxel_space.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs)
        proj = np.rint(image_points[:, 0, :]).astype(int)  # Shape: (num_voxels, 2)

        #valid projected points 
        valid = (
            (proj[:, 0] >= 0) & (proj[:, 0] < mask.shape[1]) &
            (proj[:, 1] >= 0) & (proj[:, 1] < mask.shape[0])
        )

        #determine if the corresponding mask pixel is foreground (255) via votes
        vote_contrib = np.zeros(num_voxels, dtype=int)
        vote_contrib[valid] = (mask[proj[valid, 1], proj[valid, 0]] == 255).astype(int) 
        votes += vote_contrib
        
    voxel_on = votes >= cam_votes
    return voxel_on




