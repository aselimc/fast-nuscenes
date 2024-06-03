from typing import Any, Dict, List, Tuple
from numpy.typing import ArrayLike

import os
import numpy as np
from matplotlib import pyplot as plt


def project_labels_to_images(labels: ArrayLike,
                             projection_metadata: Dict[str, Any],
                             images: Dict[str, Any],
                             cameras: List,
                             output_path: str):
    """
    Projects the labels onto the images of the given cameras
    Args:
    - labels: Labels for the pointcloud [N, 3]
    - projection_metadata: Dictionary containing the projection metadata
    - images: Dictionary containing the images
    - cameras: List of camera names
    - output_path: Directory to save the images
    """

    assert labels.shape[1] == 3, 'Labels should be in the shape [N, 3] aka colors'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axs = plt.subplots(len(cameras), 1, figsize=(16, len(cameras)*16))
    for i, cam in enumerate(cameras):
        image = np.array(images[cam])
        axs[i].imshow(image)
        points = projection_metadata['camera_projections'][cam]['points']
        # get the indices of the points that are within the image
        indices = projection_metadata['camera_projections'][cam]['indices']
        # get the colors of the points that are within the image
        colors = labels[indices]
        # project the colors onto the image
        axs[i].scatter(points[:, 0], points[:, 1], c=colors, s=5)
    # remove white space
    fig.tight_layout()
    fig.savefig(output_path)
    fig.close()
