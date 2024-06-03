from typing import Dict, Any, List, Tuple, Optional
from numpy.typing import ArrayLike

import numpy as np
import os.path as osp
from PIL import Image
from nuscenes.nuscenes import NuScenes as NuSc
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

from fast_nuscenes.utils import lidar_to_cam
from fast_nuscenes.utils import LabelMapper
from fast_nuscenes.visualizations.meshlab import MeshlabInf, write


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
        - ann_type: Annotation type ('panoptic', 'semantic') (Optional, default: None)
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
        self.annotation_type = ann_type

        self.tokens = []
        self.cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.lidar = 'LIDAR_TOP'
        self.radars = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self._get_tokens()
        self.mesh = MeshlabInf()
        if ann_type is not None:
            self.mapper = LabelMapper(self, ann_type)
        if ann_type == 'semantic':
            from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors
            self.colors = colormap_to_colors(colormap=self.colormap, name2idx=self.lidarseg_name2idx_mapping)
        elif ann_type == 'panoptic':
            from nuscenes.panoptic.panoptic_utils import generate_panoptic_colors
            self.colors = generate_panoptic_colors(colormap=self.colormap, name2idx=self.lidarseg_name2idx_mapping)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Iterator to get the data for the given token.
        
        Args:
            - idx: Index of the token

        Returns:
        {
            'sample_metadata': Dict[str, Any],
            'scene_metadata': Dict[str, Any],
            'lidar_metadata': Dict[str, Any],
            'camera_metadatas': Dict[str, Dict[str, Any]],
            'projection_metadata': Dict[str, Any],
            'images': Dict[str, PIL.Image],
            'pointcloud': PointCloud,
            'annotations': PointCloud or None,
        }        
        """
        out = {}
        token = self.tokens[idx]
        sample_metadata = self.get('sample', token)
        lidar_metadata = self.get('sample_data', sample_metadata['data']['LIDAR_TOP'])
        scene_metadata = self.get('scene', sample_metadata['scene_token'])
        camera_metadatas = {camera: self.get('sample_data',
                                             sample_metadata['data'][camera]) for camera in self.cameras}
        pointcloud, ring_indices = self._get_lidar(lidar_metadata)
        annotations = self._get_annotations(lidar_metadata)
        projection_metadata = lidar_to_cam(self, pointcloud, self.cameras, lidar_metadata, camera_metadatas)

        out.update({
            'sample_metadata': sample_metadata,
            'scene_metadata': scene_metadata,
            'lidar_metadata': lidar_metadata,
            'camera_metadatas': camera_metadatas,
            'projection_metadata': projection_metadata,
            'images': self._get_images(camera_metadatas),
            'pointcloud': pointcloud,
            'ring_indices': ring_indices,
            'annotations': annotations
        })
        return out

    def visualize(self, output_path: str, pointcloud: ArrayLike, colors: Optional[ArrayLike] = None):
        """
        Visualizes the pointcloud using Meshlab
        
        Args:
        - output_path: Path to save the visualization
        - pointcloud: Pointcloud to visualize [N, 3]
        - colors: Colors for the pointcloud [N, 3] (Optional, default: None)
        """

        # check if the pointcloud format in the path is .obj
        assert output_path.endswith('.obj'), 'Output path should be .obj file'

        if colors is not None:
            write(self.mesh, points=pointcloud, colors=colors, path=output_path)
        else:
            write(self.mesh, points=pointcloud, path=output_path)

    def get_colors(self, labels: ArrayLike) -> ArrayLike:
        """
        Gets the colors for the given labels
        
        Args:
        - labels: Labels for the pointcloud [N,]
        Returns:
        - colors: Colors for the labels [N, 3]
        """

        return self.colors[labels]

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
    
    def _get_annotations(self, lidar_metadata: Dict[str, Any]) -> ArrayLike:
        """
        Gets the annotations for the given lidar metadata
        
        Args:
        - lidar_metadata: Lidar metadata dictionary
        Returns:
        - annotations: Annotations for the pointcloud [N, 5]
        """

        if self.annotation_type == 'panoptic':
            panoptic_filename = self.get('panoptic', lidar_metadata['token'])['filename']
            path = osp.join(self.dataroot, panoptic_filename)
            anns = load_bin_file(path, type='panoptic')
            anns = self.mapper.convert_label(anns)
        elif self.annotation_type == 'semantic':
            semantic_filename = self.get('lidarseg', lidar_metadata['token'])['filename']
            path = osp.join(self.dataroot, semantic_filename)
            anns = load_bin_file(path, type='lidarseg')
            anns = self.mapper.convert_label(anns)
        else:
            anns = None
        return anns
    
    def _get_images(self, camera_metadatas: Dict[str, Any]) -> List:
        """
        Gets the images for the given camera metadatas
        
        Args:
        - camera_metadatas: Camera metadata dictionary
        Returns:
        - images: Images for the camera
        """

        images = {}
        for cam, metadata in camera_metadatas.items():
            path = osp.join(self.dataroot, metadata['filename'])
            images[cam] = Image.open(path)
        return images
