import lightning.pytorch as pl
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad, ASGD
from torch.optim.lr_scheduler import CyclicLR, PolynomialLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, StepLR, CosineAnnealingLR
#from lib.losses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from lib.datasets.cityscapes import cityscapes_color_map
from torchvision.utils import draw_segmentation_masks
from torchmetrics import JaccardIndex
import os

_EVAL_IDS =   [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33, 0] # MAP VOID CLASS TO 0 -> TOTAL BLACK 
_TRAIN_IDS =  [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]


class SegmentationModule(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 num_classes: int,
                 train_config: dict = None,
                 prediction_dir: str = None,
                 model_type: str = None,
                 model_name: str = None
                 ) -> None:        
        super().__init__()
        self.save_hyperparameters(ignore='model')
        
        self.model = model
        self.num_classes = num_classes
        
        loss_str = train_config.get('loss', 'CrossEntropy')
        self.loss = self.get_loss(loss_str)
        
        self.optimizer_config = train_config.get('optimizer')
        self.lr_schedule_config = train_config.get('lr_schedule')
        self.batch_size = train_config.get('batch_size')
        
        self.grayscale_pred_save_path = os.path.join(prediction_dir, 'predictions', model_type, model_name, 'test', 'grayscale')
        self.rgb_pred_save_path = os.path.join(prediction_dir, 'predictions', model_type, model_name, 'test', 'rgb')
        
        # Metrics
        self.train_mean_iou = JaccardIndex(task='multiclass', 
                                           num_classes=num_classes,
                                           average='macro',
                                           ignore_index=19)
        
        self.val_mean_iou = JaccardIndex(task='multiclass', 
                                         num_classes=num_classes,
                                         average='micro',
                                         ignore_index=19)
        #self.example_input_array = torch.Tensor(4, 3, 1024, 2048)
        
        
    def get_lr_schedule(self, optimizer):
        lr = self.optimizer_config.get('learnin_rate', 1e-3)
        schedule = self.lr_schedule_config.get('name')
        
        # num of steps in cyclic lr should be cycle_epochs * steps_per_epoch
        # steps_per_epoch is defined depended on the length of the dataset
        # so maybe define the dataset inside the Lightning Module using 
        # the DataModule object
        
        if schedule in ['Polynomial', 'PolynomialLr', 'PolynomialLR', 'polynomial']:
            decay_epochs = self.lr_schedule_config.get('decay_epochs')
            power = self.lr_schedule_config.get('power')
            lr_schedule = PolynomialLR(
                optimizer=optimizer,
                total_iters=decay_epochs, #*steps_per_epoch,
                power=power,
                verbose=True
            )
            
        elif schedule in ['CyclicLR', 'Cyclic', 'CyclicLr', 'cyclic']:
            lr_schedule = CyclicLR(
                optimizer = optimizer,
                base_lr = lr,
                max_lr = self.lr_schedule_config.get('max_lr', 1e-2),
                # step_size_up=
                # step_size_down=
                gamma = self.lr_schedule_config.get('gamma', 1.0),
                verbose=  True
            )

        return lr_schedule
    
    
    def get_loss(self, loss: str):
        if loss in ['CrossEntropy', 'CrossEntropyLoss', 'crossentropy']:
            loss_fn = nn.CrossEntropyLoss()
        # elif loss in ['Dice, DiceLoss']:
        #     loss_fn = DiceLoss()
        # elif loss in ['Hybrid', 'HybridLoss']:
        #     loss_fn = HybridLoss()
        # elif loss in ['rmi', 'RMI', 'RmiLoss', 'RMILoss']:
        #     loss_fn = RMILoss()
        return loss_fn
    
    def configure_optimizers(self):
        optimizer_name = self.optimizer_config.get('name', 'Adam')
        lr = self.optimizer_config.get('learnin_rate', 1e-3)
        weight_decay = self.optimizer_config.get('weight_decay', 0)
        momentum = self.optimizer_config.get('momentum', 0)
        
        optimizer_dict = {
            'Adam' : Adam(params=self.model.parameters(),
                          lr=lr,
                          weight_decay=weight_decay),
            'Adadelta' : Adadelta(params=self.model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay),
            'SGD' : SGD(params=self.model.parameters(),
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
        }

        optimizer = optimizer_dict[optimizer_name]
        lr_schedule = self.get_lr_schedule(optimizer)
        
        if lr_schedule is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': self.get_lr_schedule(optimizer)
            }
            
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    
    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        pred = self.model(input)['out']
        loss = self.loss(pred, target)
        self.train_mean_iou(torch.argmax(pred, dim=1), torch.argmax(target, dim=1))
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train_Mean_IoU', self.train_mean_iou, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        input, target = val_batch
        pred = self.model(input)['out']
        loss = self.loss(pred, target)
        self.val_mean_iou(torch.argmax(pred, dim=1), torch.argmax(target, dim=1))
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('val_Mean_IoU', self.val_mean_iou, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    
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
            pred = predictions[idx].to(device='cpu')
            input_img = input[idx].to(device='cpu')
            #print(pred.shape)
            # save grayscale predictions
            grayscale_img = Image.fromarray(pred.numpy())
            grayscale_img.save(os.path.join(self.grayscale_pred_save_path, filename))
            
            # Draw segmentation mask on top of original image
            boolean_masks = pred == torch.arange(34)[:, None, None]
            overlayed_mask = draw_segmentation_masks(input_img.to(dtype=torch.uint8), 
                                                     boolean_masks, 
                                                     alpha=0.4, 
                                                     colors=list(cityscapes_color_map.values()))
            
            #torchvision.utils.save_image(overlayed_mask.to(dtype=torch.uint8), f'{rgb_path}/{filename}')
            overlayed_mask_img = Image.fromarray(overlayed_mask.permute(1, 2, 0).to('cpu', torch.uint8).numpy(), mode='RGB')
            #overlayed_mask_img = F.to_pil_image(overlayed_mask, mode='RGB')
            overlayed_mask_img.save(os.path.join(self.rgb_pred_save_path, filename))