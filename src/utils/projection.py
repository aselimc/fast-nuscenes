from typing import Dict, Any, Tuple, List

import numpy as np
from copy import deepcopy

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def lidar_to_cam(nusc: NuScenes,
                 pcd: PointCloud,
                 cameras: List[str],
                 lidar_metadata: Dict[str, Any], 
                 camera_metadatas: Dict[str, Any],
                 normalize: bool = True,
                 threshold: float = 1.0,
                 ):
    """
    Projects the lidar pointcloud into the image plane of the given cameras
    
    Args:
    - nusc: NuScenes object
    - pcd: PointCloud object
    - cameras: List of camera names
    - lidar_metadata: Lidar metadata dictionary
    - camera_metadatas: Dictionary of camera metadata dictionaries
    - normalize: Whether to normalize the points
    - threshold: Minimum depth value for a point to be considered
    
    Returns:
    - projection_metadata: Dictionary containing the projection metadata
    """

    projection_metadata = {}
    pcd, sensor_to_ego, ego_to_global = transform_pointcloud_global_frame(nusc, deepcopy(pcd), lidar_metadata)
    projection_metadata.update({'sensor_to_ego': sensor_to_ego, 'ego_to_global': ego_to_global})
    
    camera_projections_metadata = {}
    for camera in cameras:
        camera_metadata = camera_metadatas[camera]
        pcd, global_to_ego_image, ego_to_camera, intrinsic = transform_pointcloud_camera_frame(nusc, deepcopy(pcd), camera_metadata)
        camera_projections_metadata[camera] = {'global_to_ego_image': global_to_ego_image, 'ego_to_camera': ego_to_camera}
        imsize = (camera_metadata['width'], camera_metadata['height'])
        indices, points = project_into_image(pcd, intrinsic, imsize, normalize, threshold)
        camera_projections_metadata[camera].update({'indices': indices, 'points': points})
    
    projection_metadata.update({'camera_projections': camera_projections_metadata})
    return projection_metadata


def transform_pointcloud_global_frame(nusc: NuScenes,
                                      pcd: PointCloud,
                                      lidar_metadata: Dict[str, Any]
                                      ) -> PointCloud:
    """
    Transforms the pointcloud from the ego vehicle frame to the global frame 
    for the given timestamp.

    Args:
    - nusc: NuScenes object
    - pcd: PointCloud object
    - lidar_metadata: Lidar metadata dictionary
    
    Returns:
    - pcd: PointCloud object in the global frame
    - sensor_to_ego: Transformation matrix from sensor to ego vehicle frame
    - ego_to_global: Transformation matrix from ego vehicle to global frame
    """

    # Transform pointcloud to ego vehicle frame (for that timestamp)
    calibrated_sensor = nusc.get('calibrated_sensor', lidar_metadata['calibrated_sensor_token'])
    sensor_to_ego = transform_matrix(translation=calibrated_sensor['translation'],
                                     rotation=calibrated_sensor['rotation'],
                                     inverse=False)
    pcd.transform(sensor_to_ego)

    # Transform from ego to the global frame
    ego_pose = nusc.get('ego_pose', lidar_metadata['ego_pose_token'])
    ego_to_global = transform_matrix(translation=ego_pose['translation'],
                                     rotation=ego_pose['rotation'],
                                     inverse=False)
    pcd.transform(ego_to_global)

    return pcd, sensor_to_ego, ego_to_global


def transform_pointcloud_camera_frame(nusc: NuScenes,
                      pcd: PointCloud,
                      camera_metadata: Dict[str, Any]
                      ) -> PointCloud:
    """
    Transforms the pointcloud from the global frame to the camera frame
    
    Args:
    - nusc: NuScenes object
    - pcd: PointCloud object in the global frame
    - camera_metadata: Camera metadata dictionary
    
    Returns:
    - pcd: PointCloud object in the camera frame
    - global_to_ego_image: Transformation matrix from global to ego vehicle frame (for image timestamp)
    - ego_to_camera: Transformation matrix from ego vehicle to camera frame
    - intrinsic: Intrinsic camera matrix
    """
    
    # Get the height and width of the image
    H, W = camera_metadata['height'], camera_metadata['width']
    
    assert H == 900 and W == 1600, "Camera image size is not 1600x900, resized projection not supported!"
    
    # Transform from global to ego vehicle frame (but for image timestamp)
    camera_ego_pose = nusc.get('ego_pose', camera_metadata['ego_pose_token'])
    global_to_ego_image = transform_matrix(translation=camera_ego_pose['translation'],
                                        rotation=camera_ego_pose['rotation'],
                                        inverse=True)
    pcd.transform(global_to_ego_image)

    # Transform from ego vehicle to camera frame
    calibrated_sensor = nusc.get('calibrated_sensor', camera_metadata['calibrated_sensor_token'])
    ego_to_camera = transform_matrix(translation=calibrated_sensor['translation'],
                                     rotation=calibrated_sensor['rotation'],
                                     inverse=True)
    pcd.transform(ego_to_camera)
    
    intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

    return pcd, global_to_ego_image, ego_to_camera, intrinsic


def project_into_image(pcd: PointCloud,
                       intrinsic: np.ndarray,
                       imsize: Tuple[int, int],
                       normalize: bool = True,
                       threshold: float = 1.0
                       ) -> np.ndarray:
    """
    Projects the 3D pointcloud into the 2D image plane
    
    Args:
    - pcd: PointCloud object in the camera frame
    - intrinsic: Intrinsic camera matrix
    - imsize: (width, height)
    - normalize: Whether to normalize the points
    - threshold: Minimum depth value for a point to be considered
    
    Returns:
    - indices: Indices of the points that are in the image plane
    - points: Points in the image plane
    """
    
    depths = pcd.points[2, :]
    points = view_points(pcd.points[:3, :], intrinsic, normalize=normalize)
    
    # Filter out points that are not in the image plane
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > threshold)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < imsize[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < imsize[1] - 1)
    
    indices = np.where(mask)[0]
    points = points.T[indices][:, :2].astype(np.int32)
    
    return indices, points
