import os
import torch
import torch.nn as nn
from torch.utils import tensorboard
from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad
from torch.optim.lr_scheduler import CyclicLR, PolynomialLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, StepLR, CosineAnnealingLR
#from utils.losses import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from utils.datasets import CityscapesDataset #, MapillaryDataset
from torch.utils.data import DataLoader
#from utils.eval import MeanIoU
#from utils.models import  Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus
from argparse import ArgumentParser
import yaml

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
    
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    LOGS_DIR = config['logs_dir']

    model_config = config['model']
    dataset_config = config['dataset']
    train_config = config['train_config']

    # Dataset Configuration
    DATASET = dataset_config['name']
    DATA_PATH = dataset_config['path']
    VERSION = dataset_config['version']
    NUM_TRAIN_IMAGES = dataset_config['num_train_images']
    NUM_EVAL_IMAGES = dataset_config['num_eval_images']
    CACHE = dataset_config['cache']
    CACHE_FILE = dataset_config['cache_file']
    SEED = dataset_config['seed']

    # Model Configuration
    MODEL_TYPE = model_config['architecture']
    MODEL_NAME = model_config['name']
    BACKBONE = model_config['backbone']
    UNFREEZE_AT = model_config['unfreeze_at']
    INPUT_SHAPE = model_config['input_shape']
    OUTPUT_STRIDE = model_config['output_stride']
    FILTERS = model_config['filters']
    ACTIVATION = model_config['activation']
    DROPOUT_RATE = model_config['dropout_rate']

    # Training Configuration
    PRETRAINED_WEIGHTS = model_config['pretrained_weights']
    
    BATCH_SIZE = train_config['batch_size']
    EPOCHS = train_config['epochs']
    FINAL_EPOCHS = train_config['final_epochs']
    AUGMENT = train_config['augment']
    MIXED_PRECISION = train_config['mixed_precision']
    LOSS = train_config['loss']

    optimizer_config = train_config['optimizer']
    OPTIMIZER_NAME = optimizer_config['name']
    WEIGHT_DECAY = optimizer_config['weight_decay']
    MOMENTUM = optimizer_config['momentum']
    START_LR = optimizer_config['schedule']['start_lr']
    END_LR = optimizer_config['schedule']['end_lr']
    LR_DECAY_EPOCHS = optimizer_config['schedule']['decay_epochs']
    POWER = optimizer_config['schedule']['power']

    DISTRIBUTE_STRATEGY = train_config['distribute']['strategy']
    DEVICES = train_config['distribute']['devices']

if DATASET == 'Cityscapes':
    NUM_CLASSES = 20
    IGNORE_CLASS = 19
    INPUT_SHAPE = (1024, 2048, 3)
elif DATASET == 'Mapillary':
    INPUT_SHAPE = (1024, 1856, 3)
    if VERSION == 'v1.2':
        NUM_CLASSES = 64
        IGNORE_CLASS = 63
    elif VERSION == 'v2.0':
        NUM_CLASSES = 118
        IGNORE_CLASS = 117
    else:
        raise ValueError('Version of the Mapillary Vistas dataset should be either v1.2 or v2.0!')
else:
    raise ValueError(F'{DATASET} dataset is invalid. Available Datasets are: Cityscapes, Mapillary!')

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

# ---------------------- set presicion policy ------------------------------


# --------------------------- Create Dataset stream --------------------------------
if DATASET == 'Cityscapes':
    train_ds = CityscapesDataset(root=DATA_PATH, 
                                 split='train', 
                                 mode='fine', 
                                 target_type='semantic')
    train_ds_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_ds = CityscapesDataset(root=DATA_PATH, 
                                 split='val', 
                                 mode='fine', 
                                 target_type='semantic')
    val_ds_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False)

# elif DATASET == 'Mapillary':
#     train_ds = MapillaryDataset(height=1024, width=1856,
#                                 split='training',
#                                 preprocessing=PREPROCESSING,
#                                 version=VERSION,
#                                 shuffle=True,
#                                 )
#     train_ds = train_ds.create(DATA_PATH, BATCH_SIZE, NUM_TRAIN_IMAGES, augment=False, seed=SEED)

#     val_ds = MapillaryDataset(height=1024, width=1856,
#                               split='validation',
#                               preprocessing=PREPROCESSING,
#                               version=VERSION,
#                               shuffle=False)
#     val_ds = val_ds.create(DATA_PATH, BATCH_SIZE, NUM_EVAL_IMAGES, seed=SEED)
    
# Define chechpoint paths and tensorboard object
steps_per_epoch = len(train_ds_loader)

loss_func = eval(LOSS)
loss = loss_func()

#--------------------------- Define Model -------------------------------
model = deeplabv3_mobilenet_v3_large()

optimizer_dict = {
    'Adam' : Adam(params=model.parameters,
                  lr=START_LR,
                  weight_decay=WEIGHT_DECAY),
    'Adadelta' : Adadelta(params=model.parameters,
                          lr=START_LR,
                          weight_decay=WEIGHT_DECAY),
    'SGD' : SGD(params=model.parameters,
                lr=START_LR,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY)
}

optimizer = optimizer_dict[OPTIMIZER_NAME]

lr_schedule = PolynomialLR(
    optimizer=optimizer,
    initial_learning_rate=START_LR,
    total_iters=LR_DECAY_EPOCHS*steps_per_epoch,
    power=POWER,
    verbose=True
    )

# mean_iou = MeanIoU(NUM_CLASSES, name='MeanIoU', ignore_class=None)
# mean_iou_ignore = MeanIoU(NUM_CLASSES, name='MeanIoU_ignore', ignore_class=IGNORE_CLASS)
# metrics = [mean_iou_ignore]

# model = torch.compile(model)

