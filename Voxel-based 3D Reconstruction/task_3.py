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




def reconstruct_voxels(cams, voxel_space, cam_votes=4):
    # project each voxel into each camera using tje calibration

    num_voxels = voxel_space.shape[0]
    voxel_on = np.ones(num_voxels, dtype=bool)
    calib = {}
    masks = {}

    # for each voxel count the votes from foreground masks.
    for i in range(num_voxels):
        voxel = voxel_space[i].reshape(1,3).astype(np.float64)
        votes = 0
        for cam in cams:
            calib = load_config(cam)
            mask = cv2.imread(f"data/{cam}/foreground_mask.png", cv2.IMREAD_GRAYSCALE) 
            camera_matrix, dist_coeffs, rvec, tvec = calib
            image_points, _ = cv2.projectPoints(voxel, rvec, tvec, camera_matrix, dist_coeffs)
            x, y = image_points[0][0]
            x, y = int(round(x)), int(round(y))


            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue #no vote if the 2d projection isnt covered by the camera

            if mask[y, x] == 255:
                votes += 1 #vote yes if camera sees it as foreground 
        if votes < cam_votes:
            voxel_on[i] = False
    return voxel_on



def visualize_voxels(voxel_space, voxel_mask): #dummy for now. trying to do their repo visualization
    on_voxels = voxel_space[voxel_mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(on_voxels[:,0], on_voxels[:,1], on_voxels[:,2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Voxel Reconstruction")
    plt.show()



def main():
    
    cams = [f"cam{i}" for i in range(1, 5)]
    
    #half-cubic volume
    voxel_space = create_voxel_space((0,128), (0,64), (0,128), resolution=4)
    
    voxel_mask = reconstruct_voxels(cams, voxel_space, cam_votes=2)
    visualize_voxels(voxel_space, voxel_mask)

if __name__ == "__main__":
    main()
