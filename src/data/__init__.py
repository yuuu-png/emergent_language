from .cifar10 import CIFAR10DataModule, cifar10ClassLabels
from .coco_captions import CocoCaptionsDataModule
from .coco_detection import CocoDetectionDataModule
from .coco_image import CocoImageOnlyDataModule
from .el_coco import EmergentLanguageAndCaptionDataModule
from .mnist import MNISTDataModule, mnistClassLabels
from .super_clevr_image_only import SuperCLEVRImageOnlyDataModule
from .super_clevr import SuperCLEVRDataModule

__all__ = [
    "CIFAR10DataModule",
    "cifar10ClassLabels",
    "CocoCaptionsDataModule",
    "CocoDetectionDataModule",
    "CocoImageOnlyDataModule",
    "MNISTDataModule",
    "mnistClassLabels",
    "EmergentLanguageAndCaptionDataModule",
    "SuperCLEVRImageOnlyDataModule",
    "SuperCLEVRDataModule",
]
