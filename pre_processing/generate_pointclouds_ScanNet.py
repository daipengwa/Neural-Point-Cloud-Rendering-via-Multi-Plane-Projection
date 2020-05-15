"""
ScanNet

This file help you generate point clouds from RGB_D images.

"""

from __future__ import division
import numpy as np
import os, cv2, time, math, scipy
import scipy.io as io
import argparse

def CameraParameterRead(dir):

    intrinsic_color_path = dir + 'intrinsic_color.txt'
    intrinsic_depth_path = dir + 'intrinsic_depth.txt'
    extrinsic_color_path = dir + 'extrinsic_color.txt'
    extrinsic_depth_path = dir + 'extrinsic_depth.txt'

    intrinsic_color = []
    intrinsic_depth = []
    extrinsic_color = []
    extrinsic_depth = []

    f = open(intrinsic_color_path)
    for j in range(4):
        line = f.readline()
        tmp = line.split()
        intrinsic_color.append(tmp)
    intrinsic_color = np.array(intrinsic_color, dtype=np.float32)


    f = open(intrinsic_depth_path)
    for j in range(4):
        line = f.readline()
        tmp = line.split()
        intrinsic_depth.append(tmp)
    intrinsic_depth = np.array(intrinsic_depth, dtype=np.float32)


    f = open(extrinsic_color_path)
    for j in range(4):
        line = f.readline()
        tmp = line.split()
        extrinsic_color.append(tmp)
    extrinsic_color = np.array(extrinsic_color, dtype=np.float32)


    f = open(extrinsic_depth_path)
    for j in range(4):
        line = f.readline()
        tmp = line.split()
        extrinsic_depth.append(tmp)
    extrinsic_depth = np.array(extrinsic_depth, dtype=np.float32)


    return intrinsic_color, intrinsic_depth, extrinsic_color, extrinsic_depth


def CameraPoseRead(camera_name):

    camera_pose_path = camera_name
    camera_pose = []

    f = open(camera_pose_path)
    for i in range(4):
        line = f.readline()
        tmp = line.split()
        camera_pose.append(tmp)
    camera_pose = np.array(camera_pose, dtype=np.float32)

    return camera_pose


def projection(image, depth, intrinsic_matrix, extrinsic_matrix, position_image):

    # pixel coordinates system, origin at image's top left
    height, weight, channel = image.shape
    pixel_position = np.ones([4, height*weight])
    v = np.array(position_image[0])
    u = np.array(position_image[1])
    depth = depth[v, u]
    image = np.transpose(image[v, u])

    pixel_position[0, :] = u*depth
    pixel_position[1, :] = v*depth
    pixel_position[2, :] = depth

    transform_matrix = extrinsic_matrix.dot(np.linalg.inv(intrinsic_matrix))

    # discard invalid depth position
    depth_mask = np.where(depth != 0.0)[0]
    pixel_position = pixel_position[:, depth_mask]
    position_color = image[:, depth_mask]

    #image-to-world
    position_in_world = transform_matrix.dot(pixel_position)

    return position_in_world, depth_mask, position_color


def combine(dir, num_files, remove=True):

    st = time.time()
    point_clouds_all = np.zeros([4, 0])
    point_clouds_colors_all = np.zeros([3, 0])

    for j in range(num_files):

        if not os.path.isfile(dir + 'point_clouds_%s.mat' % j):
            continue

        content = io.loadmat(dir + 'point_clouds_%s.mat' % j)
        point_clouds = content['point_clouds']
        content1 = io.loadmat(dir + 'point_clouds_colors_%s.mat' % j)
        point_clouds_colors = content1['point_clouds_color']

        point_clouds_all = np.concatenate([point_clouds_all, point_clouds], axis=1)
        point_clouds_colors_all = np.concatenate([point_clouds_colors_all, point_clouds_colors], axis=1)

    # print('load point clouds: %ss' % (time.time() - st))

    num_point = point_clouds_all.shape[1]

    target = open(out_root + "point_clouds.ply", 'w')
    target.write("ply\n")
    target.write("format ascii 1.0\n")
    target.write("element vertex %s\n" % num_point)
    target.write("property float32 x\nproperty float32 y\nproperty float32 z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
    target.write("end_header\n")

    for i in range(num_point):
        target.write('%.4f %.4f %.4f %s %s %s\n' % (
        point_clouds_all[0, i], point_clouds_all[1, i], point_clouds_all[2, i], int(point_clouds_colors_all[2, i]),
        int(point_clouds_colors_all[1, i]), int(point_clouds_colors_all[0, i])))

    target.close()

    print('%ss, Combination done!  ' % (time.time() - st))

    # remove intermediate .mat files
    if remove:

        for j in range(num_files):
            if not os.path.isfile(dir + 'point_clouds_%s.mat' % j):
                continue
            os.remove(dir + 'point_clouds_%s.mat' % j)
            os.remove(dir + 'point_clouds_colors_%s.mat' % j)

    return 0



if __name__ == '__main__':

    scene = 'scene0010_00'   # which scene used to generate point clouds.
    dir1 = '../data/ScanNet/%s/color/' % scene   # color image path
    dir2 = '../data/ScanNet/%s/depth/' % scene   # depth image path
    dir3 = '../data/ScanNet/%s/intrinsic/' % scene  # intrinsic parameter path
    dir4 = '../data/ScanNet/%s/pose/' % scene  # camera pose path
    out_root = '../pre_processing_results/ScanNet/%s/' % scene  # output path

    random_select = True  # Do not use all pixels, just randomly select.
    random_select_proportion = 1/15  # Proportion used to generate point clouds; large is better but need more resource.

    resize = True  # From 1296*968 (color image shape) to 640*480 (depth_image shape)
    target_h = 480
    target_w = 640
    position_image = np.where(np.zeros([target_h, target_w]) == 0)  # u, v

    image_list = os.listdir(dir1)
    num_image = len(image_list)  # Total number of images.
    batch_num = 200  # Due to the limitation of memory and speed. We divide whole images into small batches.

#######################################################################################################################

    intrinsic_color, intrinsic_depth, _, _ = CameraParameterRead(dir3)

    if resize:
        scale1_c = (1296 - 1) / (target_w - 1)
        scale2_c = (968 - 1) / (target_h - 1)
        scale1_d = (640 - 1) / (target_w - 1)
        scale2_d = (480 - 1) / (target_h - 1)

        intrinsic_color[0:1, :] = intrinsic_color[0:1, :] / scale1_c
        intrinsic_color[1:2, :] = intrinsic_color[1:2, :] / scale2_c
        intrinsic_depth[0:1, :] = intrinsic_depth[0:1, :] / scale1_d
        intrinsic_depth[1:2, :] = intrinsic_depth[1:2, :] / scale2_d

    cnt = 0
    CNT = 0
    point_clouds = np.empty([4, 0], dtype=np.float32)
    point_clouds_colors = np.empty([3, 0], dtype=np.float32)
    depth_masks = np.empty([1, 0], dtype=np.float32)

    st = time.time()

    for i in range(num_image):

        image_name = dir1 + '%s.jpg' % i
        depth_name = dir2 + '%s.png' % i
        camera_name = dir4 + '%s.txt' % i

        if not (os.path.isfile(image_name) and os.path.isfile(depth_name) and os.path.isfile(camera_name)):
            print('missing files!')
            cnt = cnt + 1
            continue

        image = cv2.resize(cv2.imread(image_name, -1), (target_w, target_h))
        depth = cv2.resize(cv2.imread(depth_name, -1), (target_w, target_h)) / 1000  # shift 1000
        extrinsic_matrix = CameraPoseRead(camera_name)

        if extrinsic_matrix[0, 0] < -1e10:  # avoid invalid pose (-Inf) in ScanNet
            print('invalid camera')
            cnt = cnt + 1
            continue

        if random_select:
            tmp = np.random.uniform(0, 1.0, depth.shape)
            random_map = np.ones_like(tmp)
            random_map[np.where(tmp > random_select_proportion)] = 0
            depth = depth * random_map

        # project 2D images into 3D space.
        intrinsic_matrix = intrinsic_color
        world_coordinates, depth_mask, point_clouds_color = projection(image, depth, intrinsic_matrix, extrinsic_matrix, position_image)

        point_clouds = np.concatenate((point_clouds, world_coordinates), axis=1)
        point_clouds_colors = np.concatenate((point_clouds_colors, point_clouds_color), axis=1)
        depth_masks = np.concatenate((depth_masks, np.reshape(depth_mask, (1, depth_mask.shape[0]))), axis=1)

        cnt = cnt + 1

        if (cnt == batch_num or i == (num_image - 1)):

            if not os.path.isdir(out_root):
                os.makedirs(out_root)

            io.savemat(out_root + 'point_clouds_%s.mat' % CNT, {'point_clouds': point_clouds})
            io.savemat(out_root + 'point_clouds_colors_%s.mat' % CNT, {'point_clouds_color': point_clouds_colors})

            print('one batch finished %ss' %(time.time() - st))

            cnt = 0
            CNT = CNT + 1
            point_clouds = np.empty([4, 0], dtype=np.float32)
            point_clouds_colors = np.empty([3, 0], dtype=np.float32)
            depth_masks = np.empty([1, 0], dtype=np.float32)


    print('Generate point clouds: %ss' % (time.time() - st))
    print('Combining batches ...')
    combine(dir=out_root, num_files=(num_image//batch_num + 1), remove=True)





