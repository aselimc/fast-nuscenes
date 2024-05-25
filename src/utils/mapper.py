import numpy as np
from numpy.typing import ArrayLike

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.panoptic.utils import PanopticClassMapper
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper


class LabelMapper(PanopticClassMapper, LidarsegClassMapper):
    def __init__(self, nusc: NuScenes, mode: str = 'panoptic'):
        """
        A wrapper around the PanopticClassMapper class from the nuscenes-devkit
        to provide additional functionality, where panoptic labels can be converted
        with instance information in them into coarse semantic labels
        
        Example: 4001 (general class pedestrian with instance) -> 7001 (coarse class person with instance)
        
        Args:
        - nusc: NuScenes object
        - mode: Mode of the label mapper. Can be 'panoptic' or 'semantic'
        """

        assert mode in ['panoptic', 'semantic'], 'Invalid mode'
        self.mode = mode

        if mode == 'panoptic':
            PanopticClassMapper.__init__(self, nusc)
        elif mode == 'semantic':
            LidarsegClassMapper.__init__(self, nusc)
    
    def convert_label(self, points_label: ArrayLike) -> ArrayLike:
        """
        Converts the given points label to the desired format
        
        Args:
        - points_label: Array containing the labels
        
        Returns:
        - converted_label: Array containing the converted labels
        """
        if self.mode == 'semantic':
            return super().convert_label(points_label)
        else:
            # We need to convert semantic labels and 
            # add panoptic information again
            
            # first, seperate the semantic and instance labels
            fine2coarse = self.fine_idx_2_coarse_idx_mapping
            stuff_labels = super().convert_label(points_label//1000)
            points_label = np.where(points_label % 1000 == 0, stuff_labels, points_label)
            instances = np.unique(points_label)
            for instance in instances:
                if instance % 1000 == 0:
                    continue
                instance_label = instance % 1000
                semantic_label = instance // 1000
                coarse_label = fine2coarse[semantic_label]
                panoptic_label = coarse_label * 1000 + instance_label
                points_label = np.where(points_label == instance, panoptic_label, points_label)
            return points_label
