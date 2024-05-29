from typing import Dict, Any, List, Tuple
from numpy.typing import ArrayLike

import numpy as np
import os.path as osp
from nuscenes.nuscenes import NuScenes as NuSc
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

from src.utils import lidar_to_cam


class NuScenes(NuSc):
    def __init__(self,
                 mode: str,
                 dataroot: str,
                 version: str,
                 verbose: bool = True,
                 nusc: NuSc = None,
                 scene: str = None,
                 ann_type: str = None):
        """
        Wrapper around the NuScenes class from the nuscenes-devkit to provide additional functionality
        
        Args:
        - mode: Dataset mode ('train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track')
        - nusc: NuScenes object
        - dataroot: Path to the dataset
        - version: Dataset version
        - verbose: Whether to print messages (Optional, default: True)
        - nusc: NuScenes object (Optional, default: None)
        - scene: Scene name (Optional, default: None)
        """
        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'], 'Invalid version'
        assert mode in ['train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track'], 'Invalid mode'
        assert ann_type in [None, 'panoptic', 'semantic'], 'Invalid annotation type'
        
        if nusc is not None:
            self = nusc
        else:
            super().__init__(dataroot=dataroot, version=version, verbose=verbose)

        self.mode = mode
        self.scene_name = scene

        self.tokens = []
        self.cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.lidar = 'LIDAR_TOP'
        self.radars = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self._get_tokens()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Iterator to get the data for the given token.
        
        Args:
            - idx: Index of the token

        Returns:
        {
            'sample_metadata': Dict[str, Any],
            'lidar_metadata': Dict[str, Any],
            'camera_metadatas': Dict[str, Dict[str, Any]],
            'projection_metadata': Dict[str, Any],
            'images': Dict[str, PIL.Image],
            'pointcloud': PointCloud,
            'panoptic_labels' or 'semantic_labels': PointCloud (Optional),
        }        
        """
        out = {}
        token = self.tokens[idx]
        sample_metadata = self.get('sample', token)
        lidar_metadata = self.get('sample_data', sample_metadata['data']['LIDAR_TOP'])
        camera_metadatas = {camera: self.get('sample_data', sample_metadata['data'][camera]) for camera in self.cameras}
        pcd, ring_indices = self._get_lidar(lidar_metadata)
        projection_metadata = lidar_to_cam(self, pcd, self.cameras, lidar_metadata, camera_metadatas)
        

    def __len__(self) -> int:
        return len(self.tokens)

    def _get_tokens(self) -> None:
        """
        Collects the sample tokens based on the mode and scene name. If scene name is not provided,
        it collects the tokens for the scenes in the mode
        """

        if self.scene_name is not None:
            scene = self._get_scenes([self.scene_name])
            if scene is not None:
                self.tokens = scene[0]['token']
            else:
                raise ValueError('Scene not found')
        else:
            scenes = self._get_scenes(create_splits_scenes()[self.mode])
            for scene in scenes:
                sample_token = scene['first_sample_token']
                while sample_token:
                    self.tokens.append(sample_token)
                    sample_token = self.get('sample', sample_token)['next']

    def _get_scenes(self, scene_names: List[str]) -> List[Dict[str, Any]]:
        """
        Finds the scenes with the given names
        
        Args:
        - scene_names: List of scene names
        Returns:
        - scenes: List of scenes
        """

        scenes = []
        counts = 0
        for scene in self.scene:
            if scene['name'] in scene_names:
                scenes.append(scene)
                counts += 1
            if counts == len(scene_names):
                return scenes
        return None

    def _get_lidar(self, lidar_metadata: Dict[str, Any]) -> Tuple[LidarPointCloud, ArrayLike]:
        """
        Gets the lidar pointcloud for the given metadata
        
        Args:
        - lidar_metadata: Lidar metadata dictionary
        Returns:
        - pcd: Lidar pointcloud object
        - ring_indices: Ring indices of the points [N,]
        """
        path = osp.join(self.dataroot, lidar_metadata['filename'])
        ring_indices = np.fromfile(path, dtype=np.float32).reshape((-1, 5))[:, -1]
        return LidarPointCloud.from_file(path), ring_indices