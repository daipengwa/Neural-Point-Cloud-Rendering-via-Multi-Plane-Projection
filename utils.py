import os,time, cv2
import numpy as np
import scipy.io as io

def prepare_data_ScanNet(dir1, dir2, dir3, dir4, num_image):

    image_names = []
    index_names = []
    index_names_1 = []
    camera_names = []

    for i in range(num_image):
        image_names.append(dir1 + '%s.jpg' % i)
        camera_names.append(dir2 + '%s.txt' % i)
        index_names.append(dir3 + '%s_compressed.npz' % i)
        index_names_1.append(dir4 + '%s_weight.npz' % i)


    image_names_train = []
    image_names_test = []
    index_names_train = []
    index_names_test = []
    camera_names_train = []
    camera_names_test = []
    index_names_1_train = []
    index_names_1_test = []

    flag = False
    for i in range(100):

        left = int(20 + 100*i)
        right = int(80 + 100*i)-1

        if left > len(image_names):
            break

        if right > len(image_names):
            right = len(image_names)
            flag = True

        image_names_train = image_names_train + image_names[left:right]
        image_names_test = image_names_test + image_names[int(i*100 - 1):int(i*100)]
        index_names_train = index_names_train + index_names[left:right]
        index_names_test = index_names_test + index_names[int(i*100 - 1):int(i*100)]
        camera_names_train = camera_names_train + camera_names[left:right]
        camera_names_test = camera_names_test + camera_names[int(i*100 - 1):int(i*100)]
        index_names_1_train = index_names_1_train + index_names_1[left:right]
        index_names_1_test = index_names_1_test + index_names_1[int(i*100 - 1):int(i*100)]

        if flag:
            break

    return image_names_train, index_names_train, camera_names_train, index_names_1_train, image_names_test, index_names_test, camera_names_test, index_names_1_test


def prepare_data_matterport(dir1, dir2, dir3, dir4):

    image_names = []
    index_names = []
    index_names_1 = []
    extrinsics = []
    intrinsics = []
    parameter_file = []

    for root, _, fname in os.walk(dir3):
        parameter_file.append(os.path.join(dir3, fname[0]))

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
            image_names.append(dir1 + temp[2])

    for i in range(len(image_names)):
        index_names.append(dir2 + '%s_compressed.npz' % i)
        index_names_1.append(dir4 + '%s_weight.npz' % i)

    image_names_train = []
    image_names_test = []
    index_names_train = []
    index_names_test = []
    camera_names_train = []
    camera_names_test = []
    index_names_1_train = []
    index_names_1_test = []

    flag = False
    for i in range(100):

        left = int(0 + 100*i)
        right = int(100 + 100*i)-1

        if left > len(image_names):
            break

        if right > len(image_names):
            right = len(image_names)
            flag = True

        image_names_train = image_names_train + image_names[left:right]
        image_names_test = image_names_test + image_names[int(i*100 - 1):int(i*100)]
        index_names_train = index_names_train + index_names[left:right]
        index_names_test = index_names_test + index_names[int(i*100 - 1):int(i*100)]
        camera_names_train = camera_names_train + extrinsics[left:right]
        camera_names_test = camera_names_test + extrinsics[int(i*100 - 1):int(i*100)]
        index_names_1_train = index_names_1_train + index_names_1[left:right]
        index_names_1_test = index_names_1_test + index_names_1[int(i*100 - 1):int(i*100)]

        if flag:
            break

    return image_names_train, index_names_train, camera_names_train, image_names_test, index_names_test, camera_names_test, index_names_1_train, index_names_1_test


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

    return np.transpose(np.array(position)), np.transpose(np.array(color))


def CameraPoseRead(dir):

    camera_pose_path = dir
    camera_pose = []

    f = open(camera_pose_path)
    for i in range(4):
        line = f.readline()
        tmp = line.split()
        camera_pose.append(tmp)
    camera_pose = np.array(camera_pose, dtype=np.float32)

    return camera_pose


def camera_parameter_read(extrinsic):

    tmp = extrinsic.split()
    tmp = list(map(float, tmp[3:]))
    extrinsic_matrix = np.reshape(np.array(tmp), [4, 4])
    extrinsic_matrix[:, [1, 2]] = extrinsic_matrix[:, [1, 2]] * (-1.0)  # camera coordinate system transformation

    return extrinsic_matrix

if __name__=='__main__':

    pass