from __future__ import division
import numpy as np
import os, cv2, time, math, scipy
import scipy.io as io

def makedataset(dir2):

    image_names = []
    depth_names = []
    intrinsics = []
    extrinsics = []

    assert os.path.isdir(dir2)
    parameter_file = []

    for root,_, fname in os.walk(dir2):
        parameter_file.append(os.path.join(dir2, fname[0]))

    file = open(parameter_file[0])

    while True:
        line = file.readline()
        if not line:
            break
        temp = line.split()
        if len(temp) == 0:
            continue
        if temp[0] == 'intrinsics_matrix':
            intrinsic_temp = line
        if temp[0] == 'scan':
            extrinsics.append(line)
            intrinsics.append(intrinsic_temp)
            image_names.append(temp[2])
            depth_names.append(temp[1])

    positions_world = np.zeros([len(extrinsics), 3])

    for i in range(len(extrinsics)):
        temp = extrinsics[i].split()
        positions_world[i, 0] = np.float32(temp[6])
        positions_world[i, 1] = np.float32(temp[10])
        positions_world[i, 2] = np.float32(temp[14])

    return image_names, depth_names, intrinsics, extrinsics, positions_world


def camera_parameter_read(intrinsic, extrinsic):

    tmp = intrinsic.split()
    fx = float(tmp[1])
    ux = float(tmp[3])
    fy = float(tmp[5])
    uy = float(tmp[6])
    intrinsic_matrix = np.array([[fx, 0, ux, 0], [0, fy, 1024 - uy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    tmp = extrinsic.split()
    tmp = list(map(float, tmp[3:]))
    extrinsic_matrix = np.reshape(np.array(tmp), [4, 4])
    extrinsic_matrix[:, [1, 2]] = extrinsic_matrix[:, [1, 2]] * (-1.0)

    return intrinsic_matrix, extrinsic_matrix


def projection(image, depth, intrinsic_matrix, extrinsic_matrix, position_image):

    # pixel coordinates system, origin at image's top left
    height, weight, channel = image.shape
    pixel_position = np.ones([4, height*weight])
    v = np.array(position_image[0])
    u = np.array(position_image[1])
    depth = depth[v, u]
    image = np.transpose(image[v, u])

    pixel_position[0,:] = u*depth
    pixel_position[1,:] = v*depth
    pixel_position[2,:] = depth

    transform_matrix = extrinsic_matrix.dot(np.linalg.inv(intrinsic_matrix))

    # discard invalid depth position
    depth_mask = np.where(depth != 0.0)[0]
    pixel_position = pixel_position[:, depth_mask]
    position_color = image[:, depth_mask]

    # image-to-world coordinates system
    position_in_world = transform_matrix.dot(pixel_position)

    return position_in_world, depth_mask, position_color


def combine(dir, num_files, remove=True):

    st = time.time()
    point_clouds_o = np.zeros([4, 0])
    point_clouds_colors_o = np.zeros([3, 0])

    for j in range(num_files):

        content = io.loadmat(dir + 'point_clouds_%s.mat' % j)
        point_clouds = content['point_clouds']
        content1 = io.loadmat(dir + 'point_clouds_colors_%s.mat' % j)
        point_clouds_colors = content1['point_clouds_color']

        point_clouds_o = np.concatenate([point_clouds_o, point_clouds], axis=1)
        point_clouds_colors_o = np.concatenate([point_clouds_colors_o, point_clouds_colors], axis=1)

    # print('load point clouds: %ss' % (time.time() - st))

    num_point = point_clouds_o.shape[1]

    target = open(out_root + "point_clouds.ply", 'w')
    target.write("ply\n")
    target.write("format ascii 1.0\n")
    target.write("element vertex %s\n" % num_point)
    target.write("property float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
    target.write("end_header\n")

    for i in range(num_point):
        target.write('%.4f %.4f %.4f %s %s %s\n' % (
        point_clouds_o[0, i], point_clouds_o[1, i], point_clouds_o[2, i], int(point_clouds_colors_o[2, i]),
        int(point_clouds_colors_o[1, i]), int(point_clouds_colors_o[0, i])))

    target.close()

    print('combination done: %ss' %(time.time() - st))

    # remove intermediate .mat files
    if remove:

        for j in range(num_files):
            if not os.path.isfile(dir + 'point_clouds_%s.mat' % j):
                continue
            os.remove(dir + 'point_clouds_%s.mat' % j)
            os.remove(dir + 'point_clouds_colors_%s.mat' % j)

    return 0


if __name__ == '__main__':

    # using undistorted parameters and images
    scene = '29hnd4uzFmX'   # scene name
    dir1 = '../data/Matterport3D/%s/undistorted_color_images/' % scene  # color images path
    dir2 = '../data/Matterport3D/%s/undistorted_camera_parameters/' % scene  # camera parameters path
    dir3 = '../data/Matterport3D/%s/undistorted_depth_images/' % scene  # depth images pth
    out_root = '../pre_processing_results/Matterport3D/%s/' % scene  # output path

    image_names_all, depth_names_all, intrinsics_all, extrinsics_all, camera_positions = makedataset(dir2)

    target_h = 1024
    target_w = 1280
    position_image = np.where(np.zeros([target_h, target_w]) == 0)  # u, v

    num_image = len(image_names_all)
    batch_num = 100

    random_select = True  # Do not use all pixels, just randomly select.
    random_select_proportion = 1/10  # Proportion of pixels used to generate point clouds; large is better but need more resource.

########################################################################################################################
    cnt = 0
    CNT = 0
    point_clouds = np.empty([4, 0], dtype=np.float32)
    point_clouds_colors = np.empty([3, 0], dtype=np.float32)
    depth_masks = np.empty([1, 0], dtype=np.float32)
    st = time.time()

    for i in range(num_image):

        image_name = dir1 + image_names_all[i]
        depth_name = dir3 + depth_names_all[i]
        intrinsic_name = intrinsics_all[i]
        extrinsic_name = extrinsics_all[i]

        if not (os.path.isfile(image_name) and os.path.isfile(depth_name)):
            print(image_name)
            print(depth_name)
            print('missing file!')
            continue

        image = cv2.imread(image_name, -1)
        depth = cv2.imread(depth_name, -1) / 4000  # shift 4000

        intrinsic_matrix, extrinsic_matrix = camera_parameter_read(intrinsic_name, extrinsic_name)


        if random_select:
            tmp = np.random.uniform(0, 1.0, depth.shape)
            random_map = np.ones_like(tmp)
            random_map[np.where(tmp > random_select_proportion)] = 0
            depth = depth * random_map

        # project 2D images into 3D space.
        global_coordinates, depth_mask, point_clouds_color = projection(image=image, depth=depth, intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix, position_image=position_image)

        point_clouds = np.concatenate((point_clouds, global_coordinates), axis=1)
        point_clouds_colors = np.concatenate((point_clouds_colors, point_clouds_color), axis=1)
        depth_masks = np.concatenate((depth_masks, np.reshape(depth_mask, (1, depth_mask.shape[0]))), axis=1)

        cnt = cnt + 1

        if (cnt == batch_num or i == (num_image - 1)):

            if not os.path.isdir(out_root):
                os.makedirs(out_root)
            io.savemat(out_root + 'point_clouds_%s.mat' % CNT, {'point_clouds': point_clouds})
            io.savemat(out_root + 'point_clouds_colors_%s.mat' % CNT, {'point_clouds_color': point_clouds_colors})

            print('one batch finished %ss' % (time.time() - st))

            cnt = 0
            CNT = CNT + 1
            point_clouds = np.empty([4, 0], dtype=np.float32)
            point_clouds_colors = np.empty([3, 0], dtype=np.float32)
            depth_masks = np.empty([1, 0], dtype=np.float32)

    print('Generate point clouds: %ss' % (time.time() - st))
    print('Combining batches ...')
    combine(dir=out_root, num_files=(num_image//batch_num + 1), remove=True)




