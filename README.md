# Fast nuScenes

Fast nuScenes is a Python library that provides a convenient wrapper for working with the nuScenes dataset. It offers easy access to labels, 2D-3D correspondences, and visualization methods, making it simple for developers to work with the nuScenes dataset.

PS: Currently working nicely for Semantic/Panoptic segmentation tasks, with utilization of LiDAR and Cameras. Support for other tasks and Radar will come soon.

## Installation

To install Fast nuScenes, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/aselimc/fast-nuscenes.git
    ```

2. Run the setup script:

    ```shell
    python setup.py install
    ```

## Usage

Once you have installed Fast nuScenes, you can start using it in your Python projects. Here's a simple example to get you started:

```python
from fast_nuscenes import NuScenes

dataroot = 'YOUR DATASET ROOT'
version = 'YOUR VERSION'
mode = 'train' # or 'val'
ann_type = 'panoptic' # or 'semantic'
verbose = True

nusc = NuScenes(mode=mode, dataroot=dataroot, version=version, verbose=verbose, ann_type=ann_type)

for data in nusc:
    # Here data will have the following structure. Further
    # details can be found in the code or tests.
    # {
    #         'sample_metadata': Dict[str, Any],
    #         'scene_metadata': Dict[str, Any],
    #         'lidar_metadata': Dict[str, Any],
    #         'camera_metadatas': Dict[str, Dict[str, Any]],
    #         'projection_metadata': Dict[str, Any],
    #         'images': Dict[str, PIL.Image],
    #         'pointcloud': PointCloud,
    #         'annotations': PointCloud or None,
    # }     

```
