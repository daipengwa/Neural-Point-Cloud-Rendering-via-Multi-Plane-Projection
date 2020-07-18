# Neural Point Cloud Rendering via Multi-Plane Projection

**Neural Point Cloud Rendering via Multi-Plane projection** (CVPR 2020)  
Peng Dai*, [Yinda Zhang*](https://www.zhangyinda.com/), [Zhuwen Li*](https://scholar.google.com/citations?user=gIBLutQAAAAJ&hl=en), [Shuaicheng Liu](http://www.liushuaicheng.org/), [Bing Zeng](https://scholar.google.com/citations?user=s-kUGYQAAAAJ&hl=en).
<br>[Paper](https://arxiv.org/abs/1912.04645.pdf), [Project_page](https://daipengwa.github.io/NeuralPointCloudRendering_ProjectPage/), [Video](https://www.youtube.com/embed/iWehgsCjZZE)


## Introduction
<img src='./images/framework.png' width=1000>
<br>
Our method is divided into two parts, the multi-plane based voxelization (left) and multi-plane rendering(right). For the first part, point clouds are re-projected into camera coordinate system to form frustum region and voxelization plus aggregation operations are adopted to generate a multi-plane 3D representation, which will be concatenated with normalized view direction and sent to render network. For the second part, the concatenated input is feed into a 3D neural render network to predict the product with 4 channels (i.e. RGB + blend weight) and the final output is generated by blending all planes. The training process is under the supervision of perceptual loss, and both network parameters and point clouds features are optimized according to the gradient.

## Environments
Tensorflow 1.10.0
<br>
Python 3.6
<br>
OpenCV

## Data downloads
Download datasets (i.e. ScanNet and Matterport 3D) into corresponding 'data/...' folders, including RGB_D images, camera parameters.

## Preprocessing
Before training, there are several steps required. And the pre-processed results will be stored in 'pre_processing_results/[Matterport3D or ScanNet]/[scene_name]/'.

### Point clouds generation
Generate point clouds files('point_clouds.ply') from registrated RGB-D images by running 

```python pre_processing/generate_pointclouds_[ScanNet or Matterport].py ```

Before that, you need to specific which scene is used in 'generate_pointclouds_[ScanNet or Matterport].py' (e.g. set "scene = 'scene0010_00'" for ScanNet) .

### Point clouds simplification
Based on generated point cloud files, point cloud simplification is adopted by running 

``` python pre_processing/pointclouds_simplification.py``` 

Also, you need to specific the 'point_clouds.ply' file generated from which dataset and scene in 'pointclouds_simplification.py' (e.g. set "scene = 'ScanNet/scene0010_00'"). And simplified point clouds will be saved in 'point_clouds_simplified.ply'. 

### Voxelization and Aggregation
In order to save training time, we voxelize and aggregate point clouds in advance by running 

```python pre_processing/voxelization_aggregation_[ScanNet or Matterport].py```

This will pre-compute voxelizaion and aggregation information for each camera and save them in 'reproject_results_32/' and 'weight_32/' respectively (default 32 planes). Also, you need to specific the scene in 'voxelization_aggregation_[ScanNet or Matterport].py' (e.g. set "scene = 'scene0010_00'" for ScanNet) . 

## Train
Download ['imagenet-vgg-verydeep-19.mat'](https://drive.google.com/file/d/1BAncAnrk2u82t-o8mprMlFWqhege_LgL/view?usp=sharing) into 'VGG_Model/'.
<br>
<br>
To train the model, just run ```python npcr_ScanNet.py``` for ScanNet and ```python npcr_Matterport3D.py``` for Matterport3D. You need to set 'is_training=True' and provide the paths of train related files (i.e. RGB images, camera parameters, simplified point cloud file, pre-processed aggregation and voxelizaton information) by specificing the scene name in 'npcr_[ScanNet or Matterport3D].py' (e.g. set "scene = 'scene0010_00'" for ScanNet).

The trained model (i.e. checkpoint files) and optimized point descriptors (i.e. 'descriptor.mat') will be saved in '[ScanNet or Matterport3D]_npcr_[scene_name]/'.
<br>

## Test
To test the model, also run ```python npcr_ScanNet.py``` for ScanNet and ```python npcr_Matterport3D.py``` for Matterport3D. You need to set 'is_training=False' and provide the paths of test related files (i.e. checkpoint files, optimized point descriptors, camera parameters, simplified point cloud file, pre-processed aggregation and voxelizaton information) by specificing the scene name in 'npcr_ [ScanNet or Matterport3D].py' (e.g. set "scene = 'scene0010_00'" for ScanNet). 

The test results will be saved in '[ScanNet or Matterport3D]_npcr_[scene_name]/Test_Result/'.

If you need the point cloud files and pretrained models, please email me(daipengwa@gmail.com) and show licenses of [ScanNet](https://github.com/ScanNet/ScanNet) and [Matterport3D](https://github.com/niessner/Matterport).

## Citation
If you use our code or method in your work, please cite the following:
```
@InProceedings{Dai_2019_neuralpointcloudrendering,
author = {Peng, Dai and Yinda, Zhang and Zhuwen, Li and Shuaicheng, Liu and Bing, Zeng},
title = {Neural Point Cloud Rendering via Multi-plane Projection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

