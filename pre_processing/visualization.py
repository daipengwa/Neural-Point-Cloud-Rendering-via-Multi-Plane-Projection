"""
Render multi-plane images into 2D image plane.
Here, point is not visualized as a depth related square, but you can try it.
"""
# from __future__ import absolute_import
import scipy.io as io
import numpy as np
import time, os, cv2
from utils import *

# which_datasets = 'Matterport3D'
which_datasets = 'ScanNet'

scene = 'ScanNet/scene0010_00'
num_image = 10  # number of images for visualization
target_w = 640
target_h = 480
num_planes = 32  # number of image planes.
reproject_file = '../pre_processing_results/%s/reproject_results_%s/' % (scene, num_planes)
point_clouds_path = '../pre_processing_results/%s/point_clouds_simplified.ply' % scene
output_dir = '../pre_processing_results/%s/visualized_%s/' % (scene, num_planes)

if which_datasets=='Matterport3D':

    scene = 'Matterport3D/29hnd4uzFmX'
    num_image = 10
    traget_w = 640
    target_h = 512
    num_planes = 32
    reproject_file = '../pre_processing_results/%s/reproject_results_%s/' % (scene, num_planes)
    point_clouds_path = '../pre_processing_results/%s/point_clouds_simplified.ply' % scene
    output_dir = '../pre_processing_results/%s/visualized_%s/' % (scene, num_planes)


point_clouds, point_clouds_colors = loadfile(point_clouds_path)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for i in range(num_image):
    if not os.path.isfile(reproject_file + '%s_compressed.npz' % i):
        print('Missing file: %s_compressed.npz' % i)
        continue

    npzfile = np.load(reproject_file + '%s_compressed.npz' % i)
    u = npzfile['u']
    v = npzfile['v']
    d = npzfile['d']
    index = npzfile['select_index']

    image, mask = render_point_clouds(target_h, target_w, u, v, d, index, point_clouds_colors, num_planes)
    cv2.imwrite(output_dir + '%s.jpg' % i, np.uint8(image))
