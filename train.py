import lightning.pytorch as pl
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad, ASGD
from torch.optim.lr_scheduler import CyclicLR, PolynomialLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, StepLR, CosineAnnealingLR
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
from lightning.pytorch.callbacks import ModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import sys
#from utils.losses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
from torch.utils.data import DataLoader
from torchsummary import summary
#from utils.eval import MeanIoU
#from utils.models import  Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
import yaml
from argparse import ArgumentParser

parser = ArgumentParser('')
parser.add_argument('--config', type=str, nargs='?')
parser.add_argument('--data_path', type=str, nargs='?')
parser.add_argument('--dataset', type=str, nargs='?', default='Cityscapes', choices=['Cityscapes', 'Mapillary'])
parser.add_argument('--model_type', type=str, nargs='?', choices=['Unet', 'Residual_Unet', 'Attention_Unet', 'Unet_plus', 'DeepLabV3plus'])
parser.add_argument('--model_name', type=str, nargs='?')
parser.add_argument('--backbone', type=str, nargs='?', default='None')
parser.add_argument('--output_stride', type=int, nargs='?', default=32)
parser.add_argument('--unfreeze_at', type=str, nargs='?')
parser.add_argument('--activation', type=str, nargs='?', default='relu')
parser.add_argument('--dropout', type=float, nargs='?', default=0.0)
parser.add_argument('--optimizer', type=str, nargs='?', default='Adam', choices=['Adam', 'Adadelta', 'Nadam', 'AdaBelief', 'AdamW', 'SGDW'])
parser.add_argument('--loss', type=str, nargs='?', default='FocalHybridLoss', choices=['DiceLoss', 'IoULoss', 'TverskyLoss', 'FocalTverskyLoss', 'HybridLoss', 'FocalHybridLoss'])
parser.add_argument('--batch_size', type=int, nargs='?', default='3')
parser.add_argument('--augment', type=bool, nargs='?', default=False)
parser.add_argument('--epochs', type=int, nargs='?', default='20')
parser.add_argument('--final_epochs', type=int, nargs='?', default='60')
args = parser.parse_args()

if args.config is None:
    # parse arguments
    print('Reading configuration from cmd args')
    DATA_PATH = args.data_path
    DATASET = args.dataset
    MODEL_TYPE = args.model_type
    MODEL_NAME = args.model_name
    BACKBONE = args.backbone
    OUTPUT_STRIDE = args.output_stride
    OPTIMIZER_NAME = args.optimizer
    UNFREEZE_AT = args.unfreeze_at
    LOSS = args.loss
    BATCH_SIZE = args.batch_size
    ACTIVATION = args.activation
    DROPOUT_RATE = args.dropout
    AUGMENT = args.augment
    EPOCHS = args.epochs
    FINAL_EPOCHS = args.final_epochs
else:
    # Read YAML file
    print('Reading configuration from config yaml')

    with open('config/Cityscapes.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # TODO: Add default values if a variable is not defined in the config file

    LOGS_DIR = config.get('logs_dir')
    model_config = config.get('model_config')
    dataset_config = config.get('dataset_config')
    train_config = config.get('train_config')

    # Dataset Configuration
    DATASET = dataset_config.get('name')
    NUM_TRAIN_BATCHES = dataset_config.get('num_train_batches', 1.0)
    NUM_EVAL_BATCHES = dataset_config.get('num_eval_batches', 1.0)
    BATCH_SIZE = dataset_config.get('batch_size') #
    SEED = dataset_config.get('seed')

    # Model Configuration
    MODEL_TYPE = model_config.get('architecture')
    MODEL_NAME = model_config.get('name')
    BACKBONE = model_config.get('backbone')
    UNFREEZE_AT = model_config.get('unfreeze_at')
    INPUT_SHAPE = model_config.get('input_shape')
    OUTPUT_STRIDE = model_config.get('output_stride')
    FILTERS = model_config.get('filters')
    ACTIVATION = model_config.get('activation')
    DROPOUT_RATE = model_config.get('dropout_rate')

    # Training Configuration
    # PRETRAINED_WEIGHTS = model_config['pretrained_weights']


    EPOCHS = train_config.get('epochs') #
    AUGMENTATION = train_config.get('augment') #
    PRECISION = str(train_config.get('precision')) #

    # Stohastic weight averaging parameters
    SWA = train_config.get('swa')
    if SWA is not None:
        SWA_LRS = SWA.get('lr', 1e-3)
        SWA_EPOCH_START = SWA.get('epoch_start', 0.7)

    DISTRIBUTE_STRATEGY = train_config.get('distribute').get('strategy')
    DEVICES = train_config.get('distribute').get('devices')

    # save the config in the hparams.yaml file 
    # with open(f'{LOGS_DIR}/my.yaml', 'w') as config_file:
    #     config = yaml.safe_dump(config, config_file)


# if DATASET == 'Cityscapes':
#     NUM_CLASSES = 20
#     IGNORE_CLASS = 19
#     INPUT_SHAPE = (1024, 2048, 3)
# elif DATASET == 'Mapillary':
#     INPUT_SHAPE = (1024, 1856, 3)
#     if VERSION == 'v1.2':
#         NUM_CLASSES = 64
#         IGNORE_CLASS = 63
#     elif VERSION == 'v2.0':
#         NUM_CLASSES = 118
#         IGNORE_CLASS = 117
#     else:
#         raise ValueError('Version of the Mapillary Vistas dataset should be either v1.2 or v2.0!')
# else:
#     raise ValueError(F'{DATASET} dataset is invalid. Available Datasets are: Cityscapes, Mapillary!')

# Define preprocessing according to the Backbone
if BACKBONE == 'None':
    PREPROCESSING = 'default'
    BACKBONE = None
elif 'ResNet' in BACKBONE:
    PREPROCESSING = 'ResNet'
    if 'V2' in BACKBONE:
        PREPROCESSING = 'ResNetV2'
elif 'EfficientNet' in BACKBONE:
    PREPROCESSING = 'EfficientNet'
elif 'EfficientNetV2' in BACKBONE:
    PREPROCESSING = 'EfficientNetV2'
elif 'MobileNet' == BACKBONE:
    PREPROCESSING = 'MobileNet'
elif 'MobileNetV2' == BACKBONE:
    PREPROCESSING = 'MobileNetV2'
elif 'MobileNetV3' in BACKBONE:
    PREPROCESSING = 'MobileNetV3'
elif 'RegNet' in BACKBONE:
    PREPROCESSING = 'RegNet'
else:
    raise ValueError(f'Enter a valid Backbone name, {BACKBONE} is invalid.')


torch.set_float32_matmul_precision(PRECISION)


model_checkpoint_path = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
model_checkpoint_callback = ModelCheckpoint(dirpath=LOGS_DIR,
                                            filename=model_checkpoint_path,
                                            save_weights_only=False,
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=True)

early_stopping_callback = EarlyStopping(patience=6,
                                        monitor='val_loss',
                                        # mode='max',
                                        min_delta=1e-6,
                                        verbose=True,
                                        strict=True,
                                        check_finite=True,
                                        log_rank_zero_only=True)

callbacks = [model_checkpoint_callback, ModelSummary(max_depth=3)]

if SWA is not None:
    swa_callback = StochasticWeightAveraging(swa_lrs=SWA_LRS,
                                         swa_epoch_start=SWA_EPOCH_START)
    callbacks.append(swa_callback)

logger = TensorBoardLogger(save_dir=f'{LOGS_DIR}/Tensorboard_logs',
                           name=f'{MODEL_TYPE}/{MODEL_NAME}',
                           log_graph=True)

#--------------------------- Define Model -------------------------------
model = SegmentationModule(
    model = deeplabv3_mobilenet_v3_large(num_classes=20),
    train_config=train_config
)

data_module = CityscapesDataModule(dataset_config)

trainer = pl.Trainer(
    accelerator='gpu',
    #strategy=DISTRIBUTE_STRATEGY
    devices=DEVICES,
    limit_train_batches=NUM_TRAIN_BATCHES,
    limit_val_batches=NUM_EVAL_BATCHES,
    max_epochs=EPOCHS,
    #precision=PRECISION,
    deterministic=False,
    callbacks=callbacks,
    default_root_dir=LOGS_DIR,
    logger=logger,
    #profiler='simple',
    #sync_batchnorm=True,
)

trainer.fit(model, datamodule=data_module)