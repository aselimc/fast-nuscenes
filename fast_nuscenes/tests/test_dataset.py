import unittest
import numpy as np

from fast_nuscenes.nuscenes import NuScenes
from fast_nuscenes.visualizations.image_space import project_labels_to_images

VERSION = 'v1.0-trainval'
DATAROOT = '/home/datasets/nuscenes'
MODE = 'val'
ANN_TYPE = 'panoptic'
VERBOSE = False

class TestNuScenes(unittest.TestCase):
    def test_iteration(self):
        nusc = NuScenes(mode=MODE, dataroot=DATAROOT, version=VERSION, verbose=VERBOSE, ann_type=ANN_TYPE)
        expected_keys = ['sample_metadata', 'lidar_metadata', 'camera_metadatas',
                         'projection_metadata', 'images', 'pointcloud', 'annotations', 'ring_indices']
        for data in nusc:
            self.assertEqual(set(data.keys()), set(expected_keys))
            labels = nusc.get_colors(data['annotations'])
            project_labels_to_images(labels, data['projection_metadata'], data['images'], nusc.cameras, 'tests/test.png')
            break

if __name__ == '__main__':
    unittest.main()
        