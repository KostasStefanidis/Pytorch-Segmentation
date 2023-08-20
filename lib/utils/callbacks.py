from typing import Literal
import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning, Callback, BasePredictionWriter
from lightning.pytorch.callbacks import GradientAccumulationScheduler
import torch
import torchvision.transforms.v2 as transformsv2
from torchvision.utils import draw_segmentation_masks
from lib.datasets.cityscapes import cityscapes_color_map
import os

_EVAL_IDS =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK 
_TRAIN_IDS =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]

class FreezeFeatureExtractor(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        
    def freeze_before_training(self, module: pl.LightningModule):
        self.freeze(module.model.backbone, train_bn=False)
    
    def finetune_function(self, 
                          module: pl.LightningModule,
                          current_epoch: int, 
                          optimizer: torch.optim
                          ):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=module.model.backbone,
                optimizer=optimizer,
                train_bn=True,
            )


class CityscapesPredictionWriter(BasePredictionWriter):
    def __init__(self,
                 logs_dir: str,
                 model_arch: str,
                 model_name: str,
                 write_interval: Literal['batch', 'epoch', 'batch_and_epoch'] = "batch"
                 ) -> None:
        super().__init__(write_interval)
        self.logs_dir = logs_dir
        self.grayscale_pred_save_path = os.path.join(logs_dir, 'predictions', model_arch, 
                                                     model_name, 'test', 'grayscale')
        self.rgb_pred_save_path = os.path.join(logs_dir, 'predictions', model_arch, 
                                               model_name, 'test', 'rgb')

    def write_on_batch_end(self, 
                           trainer, 
                           pl_module, 
                           prediction,
                           batch_indices, 
                           batch, 
                           batch_idx, 
                           dataloader_idx,
                           ):
        
        input = batch.get('image')
        filenames = list(batch.get('filename'))
        
        # Map back to Eval ids
        for train_id, eval_id in zip(reversed(_TRAIN_IDS), reversed(_EVAL_IDS)):        
            prediction = torch.where(prediction==train_id, eval_id, prediction)

        for idx, filename in enumerate(filenames):
            pred = prediction[idx].to('cpu')
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