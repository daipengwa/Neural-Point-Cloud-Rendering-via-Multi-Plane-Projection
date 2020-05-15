"""
This file is used to pre-process voxlization and aggregation weights, in order to save training time.

Re-project simplified point clouds to multi-plane, 32 planes are used.
"""
from __future__ import division
import numpy as np
import os, cv2, time, math, scipy
import scipy.io as io


def loadfile(ply_path):

    st = time.time()
    position = []
    color = []

    file = open(ply_path)
    begin = False
    while 1:
        line = file.readline().strip('\n')
        if not line:
            break
        line = line.split(' ')
        if begin:
            position.append(np.array([float(line[0]), float(line[1]), float(line[2]), float(1.0)]))
            color.append(np.array([float(line[5]), float(line[4]), float(line[3])]))  # rgb to bgr
        if line[0] == 'end_header':
            begin = True
    file.close()
    print('load ply time: %s' %(time.time() - st))
    return np.transpose(position), np.transpose(color)


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

    # tmp = intrinsics_all[i].split()
    tmp = intrinsic.split()
    fx = float(tmp[1])
    ux = float(tmp[3])
    fy = float(tmp[5])
    uy = float(tmp[6])
    intrinsic_matrix = np.array([[fx, 0, ux, 0], [0, fy, 1024 - uy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    tmp = extrinsic.split()
    tmp = list(map(float, tmp[3:]))
    extrinsic_matrix = np.reshape(np.array(tmp), [4, 4])
    extrinsic_matrix[:, [1, 2]] = extrinsic_matrix[:, [1, 2]] * (-1.0)  # Camera coordinate system transform.

    return intrinsic_matrix, extrinsic_matrix



def Voxelization(w, h, intrinsic_matrix, extrinsic_matrix, point_clouds, valid_depth_near, valid_depth_far, num_planes):

    st = time.time()
    transform_matrix = intrinsic_matrix.dot(np.linalg.inv(extrinsic_matrix))
    position_image = transform_matrix.dot(point_clouds)

    print('reproject_time: %s' %(time.time() - st))

    depth_all = position_image[2, :]
    u_all =position_image[0, :] / (depth_all+1e-10)
    v_all =position_image[1, :] / (depth_all+1e-10)

    valid_u = np.where((u_all >= 0) & (u_all <= (w-1)))
    valid_v = np.where((v_all >= 0) & (v_all <= (h-1)))
    valid_d = np.where((depth_all > valid_depth_near) & (depth_all < valid_depth_far))

    valid_position = np.intersect1d(valid_u, valid_v)
    valid_position = np.intersect1d(valid_position, valid_d)

    selected_depth = depth_all[valid_position]
    index = np.argsort(-selected_depth)  # depth large to small
    index = index[100:-50]  # in order to reduce outliers' influence during voxelization, we remove 100 furthest and 50 nearest points.

    valid_position_sorted = valid_position[index]
    valid_d_sorted = depth_all[valid_position_sorted]
    center_u_sorted = u_all[valid_position_sorted]
    center_v_soretd = v_all[valid_position_sorted]
    u_sorted = np.uint32(np.rint(center_u_sorted))
    v_sorted = np.uint32(np.rint(center_v_soretd))

    # calculate distance to grid center. Parallel distance.
    st = time.time()
    distance_sorted = np.sqrt(np.square(u_sorted - center_u_sorted) + np.square(v_sorted - center_v_soretd))
    print("calculate_distance: %s" % (time.time() - st))

    # 3D space voxelization
    num_valids = len(index)

    valid_d_min = valid_d_sorted[num_valids - 1]  # near depth plane
    valid_d_max = valid_d_sorted[0]  # far depth plane
    tmp = np.linspace(valid_d_max, valid_d_min, num_planes+1)
    up_boundary = tmp[1:]
    d_position = np.zeros([num_valids])  # points belong to which plane.

    st = time.time()
    cnt = 0
    for i in range(num_valids):
        tmp_d = valid_d_sorted[i]
        if tmp_d >= up_boundary[cnt]:
            d_position[i] = num_planes - cnt - 1
        else:
            for j in range(1, num_planes - cnt):
                cnt = cnt + 1
                if tmp_d >= up_boundary[cnt]:
                    d_position[i] = num_planes - cnt - 1
                    break
    print('split_time: %s' % (time.time() - st))

    # grouping
    groups_original = u_sorted + v_sorted*w + d_position*w*h  # groups
    groups_original_sort_index = np.argsort(groups_original)  # small to large

    groups_original_sorted = groups_original[groups_original_sort_index]
    u_sorted_1 = u_sorted[groups_original_sort_index]
    v_sorted_1 = v_sorted[groups_original_sort_index]
    d_position_sorted_1 = d_position[groups_original_sort_index]
    valid_position_sorted_1 = valid_position_sorted[groups_original_sort_index]
    distance_sorted_1 = distance_sorted[groups_original_sort_index]

    array = np.uint16(np.linspace(0, 1000, 1000, endpoint=False))  # assign points within one voxel or group a sequence index. Begin from 0. The max num in each group less than 1000.
    groups_index = np.zeros_like(valid_position_sorted_1)  # each group's start position.
    groups_each = np.zeros_like(valid_position_sorted_1)  # each point belongs to which group or voxel.
    groups_each_index = np.zeros_like(valid_position_sorted_1, dtype=np.uint16)  # each point's index/order in one group, a sequence.

    group_begin = 0
    cnt = 0

    for ii in range(num_valids):
        group_tmp = groups_original_sorted[ii]
        if (ii + 1) < num_valids:
            group_next = groups_original_sorted[ii+1]
            if not group_tmp == group_next:
                groups_each[group_begin:(ii+1)] = cnt
                groups_each_index[group_begin:(ii+1)] = array[0:(ii+1 - group_begin)]
                groups_index[cnt] = group_begin
                cnt = cnt + 1
                group_begin = ii + 1
        else:
            groups_each[group_begin:] = cnt
            groups_each_index[group_begin:] = array[0:(num_valids-group_begin)]
            groups_index[cnt] = group_begin

    groups_index = groups_index[0:(cnt+1)]
    print('group_time: %s' % (time.time() - st))

    # calculate max num of points in one group/voxel in each plane.
    split_each_max = np.zeros(num_planes, dtype=np.uint16)

    split_position = np.where((d_position_sorted_1[groups_index] - np.concatenate((np.array([0]), d_position_sorted_1[groups_index][0:-1]))) > 0)  # find split position of different planes.
    split_each_begin = np.concatenate((np.array([0]), groups_index[split_position]))  # split position based on all points, and reserve the begin position. Begin from 0.
    split_each_begin_in_group = np.concatenate((np.array([0]), split_position[0]))  # split position based on all groups, and reserve the begin position. Begin from 0.
    d_valid = d_position_sorted_1[groups_index[split_each_begin_in_group]]

    for j in range(len(split_each_begin)):

        begin = split_each_begin[j]

        if j < (len(split_each_begin_in_group) - 1):
            end = split_each_begin[j + 1]
            max_num = np.max(groups_each_index[begin:end]) + 1
            split_each_max[int(d_valid[j])] = max_num
        else:
            max_num = np.max(groups_each_index[begin:]) + 1
            split_each_max[int(d_valid[j])] = max_num

    # Be careful of data type, out of range.
    return np.uint16(u_sorted_1), np.uint16(v_sorted_1), np.uint8(d_position_sorted_1), np.uint32(valid_position_sorted_1), \
           np.uint32(groups_each), np.uint32(groups_index), np.uint16(groups_each_index), \
           np.uint32(split_each_begin), np.uint32(split_each_begin_in_group), np.uint16(split_each_max), \
           np.float16(distance_sorted_1)



def Aggregation(npzfile, intrinsic_matrix, extrinsic_matrix, point_clouds, a, b):

    select_index = npzfile['select_index']  # select_index begin with 0.
    index_in_each_group = npzfile['index_in_each_group']
    distance = npzfile['distance']

    st = time.time()
    transform_matrix = intrinsic_matrix.dot(np.linalg.inv(extrinsic_matrix))
    position_image = transform_matrix.dot(point_clouds)

    depth_all = position_image[2, :]
    depth_selected = depth_all[select_index] * 100  # x 100, m to cm.

    # distance to grid center, parallel distance
    distance = distance

    # distance to depth_min, vertical distance
    distance_1 = np.zeros(distance.shape)
    each_group_begin = np.where(index_in_each_group == 0)[0]
    num_valids = len(select_index)
    num_groups = len(each_group_begin)

    for i in range(num_groups):
        begin = each_group_begin[i]
        if (i+1) < num_groups:
            end = each_group_begin[i+1]
            distance_1[begin:end] = np.min(depth_selected[begin:end])
        else:
            end = num_valids
            distance_1[begin:end] = np.min(depth_selected[begin:end])
    distance_1 = depth_selected - distance_1
    # print(np.max(distance_1))
    # print(np.min(distance_1))


    # calculate_weight
    weight_1 = (1-distance)**a
    weight_2 = 1/(1+distance_1)**b
    weight_renew = weight_1*weight_2

    weight_average = np.float16(weight_renew)   # normalized weight

    group_begin = 0
    cnt = 1
    weight_sum = 0

    for ii in range(num_valids):
        weight_sum = weight_sum + weight_average[ii]
        if cnt < num_groups:
            if (ii+1) == each_group_begin[cnt]:
                weight_average[group_begin:(ii+1)] = weight_average[group_begin:(ii+1)] / weight_sum
                cnt = cnt + 1
                group_begin = ii+1
                weight_sum = 0
        else:
            end = num_valids
            weight_average[group_begin:end] = weight_average[group_begin:end] / np.sum(weight_average[group_begin:end])

    # print(time.time() - st)
    return np.float16(weight_average), np.float16(distance_1)


if __name__ == '__main__':

    num = 32  # number of planes.
    a = 1  # hyperparameter
    b = 1  # hyperparameter

    scene = 'Matterport3D/29hnd4uzFmX'
    dir1 = '../data/%s/undistorted_color_images/' % scene
    dir2 = '../data/%s/undistorted_camera_parameters/' % scene
    dir3 = '../data/%s/undistorted_depth_images/' % scene
    point_clouds_path = '../pre_processing_results/%s/point_clouds_simplified.ply' % scene
    output_dir = '../pre_processing_results/%s/reproject_results_%s/' % (scene, num)
    output_dir1 = '../pre_processing_results/%s/weight_%s/' % (scene, num)

    # meter, valid depth range
    near = 0.2
    far = 10

    resize = True  # From 1024*1280 to 512*640.
    target_h = 512
    target_w = 640

###################################################################################################

    image_names_all, depth_names_all, intrinsics_all, extrinsics_all, positions_world = makedataset(dir2)
    point_clouds, point_clouds_colors = loadfile(point_clouds_path)

    scale_w = (1280-1)/(target_w-1)
    scale_h = (1024-1)/(target_h-1)

    # Voxelization
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tmp = 0
    for i in range(len(image_names_all)):
        if os.path.isfile(output_dir + '%s_compressed.npz' % i):
            continue

        st = time.time()
        depth_name = dir3 + depth_names_all[i]
        depth_image = cv2.imread(depth_name, -1) / 4000  # shift 4000
        intrinsics_name = intrinsics_all[i]
        extrinsics_name = extrinsics_all[i]

        intrinsic_matrix, extrinsic_matrix = camera_parameter_read(intrinsics_name, extrinsics_name)
        if resize:
            intrinsic_matrix[0:1, :] = intrinsic_matrix[0:1, :] / scale_w
            intrinsic_matrix[1:2, :] = intrinsic_matrix[1:2, :] / scale_h

        u, v, d, index, \
        groups_each, groups_index, groups_each_index, \
        split_each_begin, split_each_begin_in_group, split_each_max,\
        distance = Voxelization(target_w, target_h, intrinsic_matrix, extrinsic_matrix, point_clouds, near, far, num)
        print(time.time() - st)

        np.savez_compressed(output_dir + '%s_compressed' % i, u=u, v=v, d=d, select_index=index,
                            group_belongs=groups_each, index_in_each_group=groups_each_index,
                            distance=distance, each_split_max_num=split_each_max)

        print('Voxelization_time: %ss' % (time.time() - st))


    # Aggregation
    if not os.path.isdir(output_dir1):
        os.makedirs(output_dir1)

    for i in range(len(image_names_all)):

        if os.path.isfile(output_dir1 + '%s_weight.npz' % i):
            continue

        st = time.time()
        depth_name = dir3 + depth_names_all[i]
        depth_image = cv2.imread(depth_name, -1) / 4000
        intrinsics_name = intrinsics_all[i]
        extrinsics_name = extrinsics_all[i]

        intrinsic_matrix, extrinsic_matrix = camera_parameter_read(intrinsics_name, extrinsics_name)

        if resize:
            intrinsic_matrix[0:1, :] = intrinsic_matrix[0:1, :] / scale_w
            intrinsic_matrix[1:2, :] = intrinsic_matrix[1:2, :] / scale_h


        if not os.path.isfile(output_dir + '%s_compressed.npz' % i):
            print('Missing voxelization information!')
            continue

        npzfile = np.load(output_dir + '%s_compressed.npz' % i)
        weight_average, distance_to_depth_min = Aggregation(npzfile, intrinsic_matrix, extrinsic_matrix, point_clouds, a, b)
        np.savez_compressed(output_dir1 + '%s_weight' % i, weight_average=weight_average, distance_to_depth_min=distance_to_depth_min)
        print('Aggregation_time: %ss' % (time.time() - st))







