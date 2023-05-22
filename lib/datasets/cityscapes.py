import torch
import torchvision
from torch import Tensor
import lightning.pytorch as pl
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, List, Optional, Callable, Any, Union
from PIL import Image
from torchvision.models import RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights
from torchvision.models import RegNet_Y_8GF_Weights, EfficientNet_V2_M_Weights
from torchvision.models import EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision import datapoints
import torchvision.transforms.v2 as transformsv2
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import get_worker_info
from lib.utils.augmentation import get_augmentations


_IGNORE_IDS = [-1,0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
_EVAL_IDS =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
_TRAIN_IDS =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18]


class EvalToTrainIds():
    def __init__(self, 
                 ignore_ids = _IGNORE_IDS, 
                 eval_ids = _EVAL_IDS, 
                 train_ids = _TRAIN_IDS
                 ):
        self.ignore_ids = ignore_ids
        self.eval_ids = eval_ids
        self.train_ids = train_ids
            
    def __call__(self, target):    
        for id in self.ignore_ids:
            target = torch.where(target==id, 34, target)
        for train_id, eval_id in zip(self.train_ids, self.eval_ids):
            target = torch.where(target==eval_id, train_id, target)
        target = torch.where(target==34, 19, target)
        return target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class OneHot():
    def __init__(self, channels) -> None:
        self.channels = channels
    
    def __call__(self, input: Tensor) -> Any:
        one_hot_output = torch.zeros(self.channels, *input.shape[1:], dtype=torch.float32)
        one_hot_output.scatter_(0, input.to(dtype=torch.int64), 1)
        return one_hot_output
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channels={self.channels})"

DEFAULT_TARGET_TRANSFORM = lambda num_classes: transformsv2.Compose(
    [
        EvalToTrainIds(),
        OneHot(num_classes)
    ]
)

DEFAULT_TRANSFORM = transformsv2.Compose(
    [
        transformsv2.ConvertImageDtype(torch.float32),
        #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

class CityscapesTestSplit(Cityscapes):
    def __init__(self, 
                 root: str, 
                 transform: Callable[..., Any] | None = None,
                 ) -> None:
        
        super().__init__(root=root,
                         transform=transform,
                         split='test', 
                         mode='fine', 
                         target_type='semantic',  
                         target_transform=None, 
                         transforms=None)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, filename): The input image to be inserted to the model for prediction and the 
            filename of that image.
        """
        image_path = self.images[index]
        image_filename = image_path.split('/')[-1]
        
        image = Image.open(image_path).convert("RGB")
        # need custon collate_fn to return filename
        
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'filename': image_filename
        }


class CityscapesDataset(Cityscapes):
    '''
    Helpler class to wrap Cityscapes with adittional funcionality.
    To be used with transforms v2. Images need to be wrapped in datapoint.Image object
    and target in datapoint.Mask
    '''
    def __init__(self, 
                 root: str, 
                 split: str = "train", 
                 mode: str = "fine", 
                 target_type: List[str] | str = "semantic", 
                 transform: Callable[..., Any] | None = None, 
                 target_transform: Callable[..., Any] | None = None, 
                 augmentations: Callable[..., Any] | None = None
                 ) -> None:
        
        super().__init__(root, split, mode, target_type, transform, target_transform)
        self.augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[datapoints.Image, datapoints.Mask]:
        
        image = Image.open(self.images[index]).convert("RGB")

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        
        # Wrap image and target in datapoints Image and Mask objects 
        image, target = datapoints.Image(image), datapoints.Mask(target)
        
        if self.augmentations is not None:
            image, target = self.augmentations(image, target)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_config: dict, 
                 augmentation_config: dict,
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 ) -> None:
        
        super().__init__()
        self.datapath = dataset_config['path']
        self.mode = dataset_config.get('mode', 'fine')
        self.target_type = dataset_config.get('type', 'semantic')
        self.num_classes = dataset_config.get('num_classes', 20)
        self.batch_size = dataset_config.get('batch_size', 3)
        self.num_workers = dataset_config.get('num_workers', 4)
        if target_transform is not None:
            self.target_transform = target_transform  
        else: 
            self.target_transform = DEFAULT_TARGET_TRANSFORM(self.num_classes)
            
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        
        if augmentation_config is not None:
            self.augmentations = get_augmentations(augmentation_config)
        else:
            self.augmentations = None
        
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_ds = CityscapesDataset(root=self.datapath, 
                                             split='train', 
                                             mode=self.mode,
                                             target_type=self.target_type,
                                             transform=self.transform,
                                             target_transform=self.target_transform,
                                             augmentations=self.augmentations)
            
            self.val_ds = CityscapesDataset(root=self.datapath, 
                                             split='val', 
                                             mode=self.mode,
                                             target_type=self.target_type,
                                             transform=self.transform,
                                             target_transform=self.target_transform)
        
        if stage == 'test':
            self.test_ds = CityscapesDataset(root=self.datapath, 
                                             split='val', 
                                             mode=self.mode,
                                             target_type=self.target_type,
                                             transform=self.transform,
                                             target_transform=self.target_transform)
        
        if stage == 'predict':
            self.predict_ds = CityscapesTestSplit(root=self.datapath,
                                                  transform=self.transform)
    
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_ds) if torch.distributed.is_initialized() else None
        return DataLoader(self.train_ds, 
                          self.batch_size, 
                          shuffle=True and sampler is None, 
                          num_workers=self.num_workers, 
                          pin_memory=True,
                          sampler=sampler)
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_ds) if torch.distributed.is_initialized() else None
        return DataLoader(self.val_ds, 
                          self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          pin_memory=True,
                          sampler=sampler)
    
    def test_dataloader(self):
        return DataLoader(self.val_ds, 
                          self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          pin_memory=True)
        
    def predict_dataloader(self):
        return DataLoader(self.test_ds, 
                          self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          pin_memory=True)


#def _worker_init_function():
    

def get_preprocessing(self, backbone, weight_version):

    preprocessing_options = {
        'default': transforms.ToTensor(), # Convert to Tensor and
        #'ResNet': resnet.preprocess_input,
        #'ResNetV2' : resnet_v2.preprocess_input,
        #'MobileNet' : mobilenet.preprocess_input,
        #'MobileNetV2' : mobilenet_v2.preprocess_input,
        'MobileNetV3' : MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms(),
        #'EfficientNet' : efficientnet.preprocess_input,
        'EfficientNetV2' : EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms(),
        'RegNet' : RegNet_Y_8GF_Weights.IMAGENET1K_V2.transforms()
    }
    return preprocessing_options[self.preprocessing]


# dictionary that contains the mapping of the class numbers to rgb color values
cityscapes_color_map =  {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64,128),
    8: (244, 35,232),
    9: (250,170,160),
    10: (230,150,140),
    11: ( 70, 70, 70),
    12: (102,102,156),
    13: (190,153,153),
    14: (180,165,180),
    15: (150,100,100),
    16: (150,120, 90),
    17: (153,153,153),
    18: (153,153,153),
    19: (250,170, 30),
    20: (220,220,  0),
    21: (107,142, 35),
    22: (152,251,152),
    23: (70,130,180),
    24: (220, 20, 60),
    25: (255,  0,  0),
    26: (0,  0,142),
    27: (0,  0, 70),
    28: (0, 60,100),
    29: (0, 60,100),
    30: (0,  0,110),
    31: (0, 80,100),
    32: (0,  0,230),
    33: (119, 11, 32)
}