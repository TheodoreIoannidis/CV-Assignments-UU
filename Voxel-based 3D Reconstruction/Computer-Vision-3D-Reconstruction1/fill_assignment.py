from skimage import measure
import trimesh

        # image_points, _ = cv2.projectPoints(voxel_space.astype(np.float64),
        #                             rvec.astype(np.float64),
        #                             tvec.astype(np.float64),
        #                             camera_matrix.astype(np.float64),
        #                             dist_coeffs.astype(np.float64))

def voxel_to_mesh(voxel_space, voxel_mask):
    """
    Convert the voxel representation into a surface mesh using Marching Cubes.
    """
    # Create a 3D volume grid filled with 0s (background)
    grid_size = (128, 64, 128)  # Adjust based on your voxel grid size
    volume = np.zeros(grid_size, dtype=np.uint8)

    # Mark "on" voxels in the volume
    for voxel in voxel_space[voxel_mask]:  # voxel_mask filters active voxels
        x, y, z = map(int, voxel)
        volume[x, y, z] = 1  # Mark voxel as occupied

    # Apply Marching Cubes to extract a surface mesh
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    # Convert to a format usable in 3D visualization
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    return mesh



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
    mesh = voxel_to_mesh(voxel_space, voxel_mask)
    mesh.export("voxel_mesh.obj")  # Save the mesh
    mesh.show()

    return data, colors, mesh