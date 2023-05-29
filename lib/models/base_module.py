import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
import torch
from lib.datasets.cityscapes import cityscapes_color_map
from lib.utils.training import get_loss, get_lr_schedule, get_optimizer
from torchvision.utils import draw_segmentation_masks
from torchmetrics import JaccardIndex
import os
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3
import torchvision.transforms.v2 as transformsv2

_EVAL_IDS =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK 
_TRAIN_IDS =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

_ALL_ARCHITECTURES = ['deeplabv3']

_MODEL_NOT_IMPLEMENTED_ERROR = lambda arch: f'''\
Architecture: {arch} is not implemented. \
Available architectures are: {_ALL_ARCHITECTURES}
'''

def get_model(architecture: str, backbone: str, num_classes: int):
    if architecture == 'deeplabv3':
        model_function_name = f'{architecture}_{backbone}'
        model_func: function = eval(model_function_name)
        model: DeepLabV3 = model_func(num_classes=num_classes)
    else:
        raise ValueError(_MODEL_NOT_IMPLEMENTED_ERROR(architecture))
    return model

class SegmentationModule(pl.LightningModule):
    def __init__(self, 
                 model_config: dict,
                 train_config: dict = None,
                 logs_dir: str = None,
                 ) -> None:
        
        super().__init__()
        self.save_hyperparameters(ignore='logs_dir')
        
        # Get all model parameters/configuration
        self.model_name = model_config.get('name')
        self.model_arch = model_config.get('architecture')
        self.model_backbone = model_config.get('backbone')
        self.weights = model_config.get('pretrained_weights')
        self.num_classes = model_config.get('num_classes', 20)
        
        self.model = get_model(self.model_arch, self.model_backbone, self.num_classes)
        
        # Get all training parameters/configuration
        loss_str = train_config.get('loss', 'CrossEntropy')
        self.loss = get_loss(loss_str)
        
        self.optimizer_config:dict = train_config.get('optimizer')
        self.lr_schedule_config:dict = train_config.get('lr_schedule')
        self.batch_size = train_config.get('batch_size')
        
        self.grayscale_pred_save_path = os.path.join(logs_dir, 'predictions', self.model_arch, 
                                                     self.model_name, 'test', 'grayscale')
        self.rgb_pred_save_path = os.path.join(logs_dir, 'predictions', self.model_arch, 
                                               self.model_name, 'test', 'rgb')
        
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
        filenames = list(predict_batch.get('filename'))
        
        predictions = self.model(input)['out']
        predictions = torch.argmax(predictions, 1)
        predictions = predictions.to(dtype=torch.uint8)

        # Map back to Eval ids
        for train_id, eval_id in zip(reversed(_TRAIN_IDS), reversed(_EVAL_IDS)):        
            predictions = torch.where(predictions==train_id, eval_id, predictions)

        for idx, filename in enumerate(filenames):
            pred = predictions[idx].to('cpu')
            img = input[idx].to('cpu')
            img = transformsv2.functional.convert_image_dtype(img, dtype=torch.uint8)
            
            # save grayscale predictions
            grayscale_img = transformsv2.functional.to_pil_image(pred)
            grayscale_img.save(f'{self.grayscale_pred_save_path}/{filename}')
            
            # Draw segmentation mask on top of original image
            boolean_masks = pred == torch.arange(34)[:, None, None]
            overlayed_mask = draw_segmentation_masks(img, 
                                                    boolean_masks, 
                                                    alpha=0.4, 
                                                    colors=list(cityscapes_color_map.values()))
            
            overlayed_mask_img = transformsv2.functional.to_pil_image(overlayed_mask)
            overlayed_mask_img.save(f'{self.rgb_pred_save_path}/{filename}')