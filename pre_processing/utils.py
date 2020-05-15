import scipy.io as io
import numpy as np
import time, os, cv2

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


def render_point_clouds(h, w, u, v, d, index, point_clouds_colors, num_planes):

    image_all = np.zeros([num_planes, h, w, 3])
    mask_all = np.zeros([num_planes, h, w, 1])
    image_all[d, v, u, :] = np.transpose(point_clouds_colors[:, index])
    mask_all[d, v, u, :] = 1

    image = np.zeros([h, w, 3])
    mask = np.zeros([h, w, 1])
    for i in range(num_planes):
        if i == 0:
            image = image_all[0, :, :, :]
            mask = mask_all[0, :, :, :]
        else:
            mask_tmp = mask_all[i, :, :, :]
            valid = mask_tmp*(1 - mask)
            image = image*mask + valid*image_all[i, :, :, :]
            mask = mask + valid

    return image, mask

if __name__=='__main__':
    pass



