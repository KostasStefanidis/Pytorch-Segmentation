import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
import torch
from lib.datasets.cityscapes import cityscapes_color_map
from lib.utils.training import get_loss, get_lr_schedule, get_optimizer
from torchvision.utils import draw_segmentation_masks
from torchmetrics import JaccardIndex
import os
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101, DeepLabV3
import torchvision.transforms.v2 as transformsv2
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights

_ALL_ARCHITECTURES = ['deeplabv3']

_MODEL_NOT_IMPLEMENTED_ERROR = lambda arch: f'''\
Architecture: {arch} is not implemented. \
Available architectures are: {_ALL_ARCHITECTURES}
'''

_ALL_BACKBONES = {
    'deeplabv3': ['mobilenet_v3_large', 'resnet50', 'resnet101']
}

_BACKBONE_NOT_IMPLEMENTED_ERROR_ = lambda arch, backbone: f'''\
Backbone: {backbone} is not implemented for {arch}. \
Available architectures are: {_ALL_BACKBONES[arch]}
'''

_BACKBONE_WEIGHTS = {
    'mobilenet_v3_large': 'MobileNet_V3_Large_Weights',
    'resnet50': 'ResNet50_Weights',
    'resnet101': 'ResNet101_Weights'
}

def get_model(architecture: str, backbone: str, weights_backbone: str, num_classes: int):
    architecture = architecture.lower()
    if architecture not in _ALL_ARCHITECTURES:
        raise ValueError(_MODEL_NOT_IMPLEMENTED_ERROR(architecture))
    
    backbone = backbone.lower()
    if backbone not in _ALL_BACKBONES[architecture]:
        raise ValueError(_BACKBONE_NOT_IMPLEMENTED_ERROR_(architecture, backbone))
    
    model_function_name = f'{architecture}_{backbone}'
    model_func: function = eval(model_function_name)
    weights_backbone = eval(f'{_BACKBONE_WEIGHTS[backbone]}.{weights_backbone.upper()}')
    
    model: DeepLabV3 = model_func(num_classes=num_classes, weights_backbone=weights_backbone)
    
    return model


class SegmentationModule(pl.LightningModule):
    def __init__(self, 
                 model_config: dict,
                 train_config: dict = None,
                 ) -> None:
        
        super().__init__()
        
        # Get all model parameters/configuration
        self.model_name = model_config.get('name')
        self.model_arch = model_config.get('architecture')
        self.model_backbone = model_config.get('backbone')
        self.weights_backbone = model_config.get('weights_backbone')
        self.weights = model_config.get('pretrained_weights')
        self.num_classes = model_config.get('num_classes', 20)
        
        self.model = get_model(self.model_arch, self.model_backbone,
                               self.weights_backbone, self.num_classes)
        
        # Get all training parameters/configuration
        loss_str = train_config.get('loss', 'CrossEntropy')
        self.loss = get_loss(loss_str)
        
        self.optimizer_config:dict = train_config.get('optimizer')
        self.lr_schedule_config:dict = train_config.get('lr_schedule')
        self.batch_size = train_config.get('batch_size')
        
        # Metrics
        self.train_mean_iou = JaccardIndex(task='multiclass', 
                                           num_classes=self.num_classes,
                                           average='macro',
                                           ignore_index=19)
        
        self.val_mean_iou = JaccardIndex(task='multiclass', 
                                         num_classes=self.num_classes,
                                         average='macro',
                                         ignore_index=19)


    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_config, self.model)
        lr_schedule = get_lr_schedule(self.lr_schedule_config, optimizer)
        
        if lr_schedule is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_schedule
            }

    
    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        pred = self.model(input)['out']
        loss = self.loss(pred, target)
        self.train_mean_iou(torch.argmax(pred, dim=1), torch.argmax(target, dim=1))
        self.log("train_loss", loss, on_epoch=True, 
                 on_step=False, sync_dist=True, prog_bar=True)
        self.log('train_Mean_IoU', self.train_mean_iou, on_epoch=True, 
                 on_step=False, sync_dist=True, prog_bar=True)
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        input, target = val_batch
        pred = self.model(input)['out']
        loss = self.loss(pred, target)
        self.val_mean_iou(torch.argmax(pred, dim=1), torch.argmax(target, dim=1))
        self.log("val_loss", loss, on_epoch=True, 
                 on_step=False, sync_dist=True, prog_bar=True)
        self.log('val_Mean_IoU', self.val_mean_iou, on_epoch=True, 
                 on_step=False, sync_dist=True, prog_bar=True)
    
    
    def predict_step(self, predict_batch: dict, batch_idx: int, dataloader_idx: int = 0):
        input = predict_batch.get('image')
        
        predictions = self.model(input)['out']
        predictions = torch.argmax(predictions, 1)
        predictions = predictions.to(dtype=torch.uint8)

        return predictions