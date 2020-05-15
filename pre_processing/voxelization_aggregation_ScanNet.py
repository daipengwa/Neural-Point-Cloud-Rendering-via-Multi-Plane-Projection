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



def preparedata(dir):

    color_names = []
    assert os.path.isdir(dir), '%s is not a valid dir' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            color_names.append(path)

    return color_names


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


def CameraPoseRead(dir, index):

    camera_pose_path = dir + '%s.txt' % index
    camera_pose = []

    f = open(camera_pose_path)
    for i in range(4):
        line = f.readline()
        tmp = line.split()
        camera_pose.append(tmp)
    camera_pose = np.array(camera_pose, dtype=np.float32)

    return camera_pose


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
    # st = time.time()
    distance_sorted = np.sqrt(np.square(u_sorted - center_u_sorted) + np.square(v_sorted - center_v_soretd))
    # print("calculate_distance: %s" % (time.time() - st))

    # 3D space voxelization. Points belong to which voxel/group.
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
    # st = time.time()
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
    # print('group_time: %s' % (time.time() - st))

    # calculate max num of points in one group/voxel in each plane.
    split_each_max = np.zeros(num_planes, dtype=np.uint16)

    split_position = np.where((d_position_sorted_1[groups_index] - np.concatenate((np.array([0]), d_position_sorted_1[groups_index][0:-1]))) > 0)  # find split position of different planes.
    split_each_begin = np.concatenate((np.array([0]), groups_index[split_position]))
    split_each_begin_in_group = np.concatenate((np.array([0]), split_position[0]))
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
    # Using np.uint32 for storage is resource consuming, maybe you can scale them(e.g. /1000), then store the scaled number.
    return np.uint16(u_sorted_1), np.uint16(v_sorted_1), np.uint8(d_position_sorted_1), np.uint32(valid_position_sorted_1), \
           np.uint32(groups_each), np.uint32(groups_index), np.uint16(groups_each_index), \
           np.uint32(split_each_begin), np.uint32(split_each_begin_in_group), np.uint16(split_each_max), \
           np.float16(distance_sorted_1)


def Aggregation(npzfile, intrinsic_matrix, extrinsic_matrix, point_clouds, a, b):

    select_index = npzfile['select_index']  # select_index begin from 0.
    index_in_each_group = npzfile['index_in_each_group']
    distance = npzfile['distance']

    # st = time.time()
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


    # calculate_weights
    weight_1 = (1-distance)**a
    weight_2 = 1/(1+distance_1)**b
    weight_renew = weight_1*weight_2

    weight_average = np.float16(weight_renew)   # normalized weights

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

    # print('Aggregation_time: %s' %(time.time() - st))
    return np.float16(weight_average), np.float16(distance_1)



if __name__ == '__main__':

    num = 32  # number of planes.
    a = 1  # hyperparameter
    b = 1  # hyperparameter

    scene = 'scene0010_00'
    dir1 = '../data/ScanNet/%s/depth/' % scene
    dir2 = '../data/ScanNet/%s/intrinsic/' % scene
    dir3 = '../data/ScanNet/%s/pose/' % scene
    dir4 = '../data/ScanNet/%s/color/' % scene
    point_clouds_path = '../pre_processing_results/ScanNet/%s/point_clouds_simplified.ply' % scene
    output_dir = '../pre_processing_results/ScanNet/%s/reproject_results_%s/' % (scene, num)
    output_dir1 = '../pre_processing_results/ScanNet/%s/weight_%s/' % (scene, num)

    # meter, valid depth range
    near = 0.2
    far = 10

    resize = True  # From 1296*968 to 640*480 (depth_image shape), keep identity with point clouds generation.
    target_h = 480
    target_w = 640

######################################################################################################################

    color_names = sorted(preparedata(dir4))
    point_clouds, point_clouds_colors = loadfile(point_clouds_path)

    intrinsic_color, intrinsic_depth, _, _ = CameraParameterRead(dir2)
    if resize:
        scale1_c = (1296 - 1) / (target_w - 1)
        scale2_c = (968 - 1) / (target_h - 1)
        scale1_d = (640 - 1) / (target_w - 1)
        scale2_d = (480 - 1) / (target_h - 1)

        intrinsic_color[0:1, :] = intrinsic_color[0:1, :] / scale1_c
        intrinsic_color[1:2, :] = intrinsic_color[1:2, :] / scale2_c
        intrinsic_depth[0:1, :] = intrinsic_depth[0:1, :] / scale1_d
        intrinsic_depth[1:2, :] = intrinsic_depth[1:2, :] / scale2_d

    intrinsic_matrix = intrinsic_color

    # Voxelization
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tmp = 0
    for i in range(len(color_names)):
        if os.path.isfile(output_dir + '%s_compressed.npz' % i):
            continue

        st = time.time()
        depth_name = dir1 + '%s.png' % i
        depth_image = cv2.imread(depth_name, -1) / 1000
        extrinsic_matrix = CameraPoseRead(dir3, i)

        if extrinsic_matrix[0, 0] < -1e6:  # avoid invalid pose -inf
            continue

        u, v, d, index, \
        groups_each, groups_index, groups_each_index, \
        split_each_begin, split_each_begin_in_group, split_each_max, \
        distance = Voxelization(target_w, target_h, intrinsic_matrix, extrinsic_matrix, point_clouds, near, far, num)

        np.savez_compressed(output_dir + '%s_compressed' % i, u=u, v=v, d=d, select_index=index,
                            group_belongs=groups_each, index_in_each_group=groups_each_index,
                            distance=distance, each_split_max_num=split_each_max)

        print('Voxelization_time: %ss' %(time.time() - st))


    # Aggregation
    if not os.path.isdir(output_dir1):
        os.makedirs(output_dir1)

    for i in range(len(color_names)):

        if os.path.isfile(output_dir1 + '%s_weight.npz' % i):
            continue

        st = time.time()
        depth_name = dir1 + '%s.png' % i
        depth_image = cv2.imread(depth_name, -1) / 1000
        extrinsic_matrix = CameraPoseRead(dir3, i)

        if extrinsic_matrix[0, 0] < -1e6:  # avoid invalid pose -inf
            continue

        if not os.path.isfile(output_dir + '%s_compressed.npz' % i):
            print('Missing voxelization information!')
            continue

        npzfile = np.load(output_dir + '%s_compressed.npz' % i)

        weight_average, distance_to_depth_min = Aggregation(npzfile, intrinsic_matrix, extrinsic_matrix, point_clouds, a, b)

        np.savez_compressed(output_dir1 + '%s_weight' % i, weight_average=weight_average, distance_to_depth_min=distance_to_depth_min)
        print('Aggregation_time: %ss' %(time.time() - st))









