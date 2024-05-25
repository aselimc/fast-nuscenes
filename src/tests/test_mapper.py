import unittest
import numpy as np
from nuscenes.nuscenes import NuScenes

from src.utils import LabelMapper


VERSION = 'v1.0-trainval'
DATAROOT = '/home/datasets/nuscenes'


class TestLabelMapper(unittest.TestCase):
    def test_convert_label(self):
        # Test the LabelMapper class
        nusc = NuScenes(version=VERSION, dataroot=DATAROOT)
        label_mapper = LabelMapper(nusc, mode='panoptic')
        
        # Test the convert_label method
        points_label = np.array([4001, 4002, 4003, 4004, 4005])
        converted_label = label_mapper.convert_label(points_label)
        expected_label = np.array([7001, 7002, 7003, 7004, 7005])
        np.testing.assert_array_equal(converted_label, expected_label)


if __name__ == '__main__':
    unittest.main()