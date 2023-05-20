import torch
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
from argparse import ArgumentParser
import yaml
import os

parser = ArgumentParser('')
parser.add_argument('--config', type=str, nargs='?')
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = yaml.safe_load(config_file)
    
    LOGS_DIR = config['logs_dir']

    dataset_config = config['dataset']
    model_config = config['model']
    inference_config = config['inference_config']
    
    DATASET = dataset_config['name']
    DATA_PATH = dataset_config['path']
    VERSION = dataset_config['version']
    
    MODEL_TYPE = model_config['architecture']
    MODEL_NAME = model_config['name']
    BACKBONE = model_config['backbone']
    
    INFERENCE_OUTPUT_STRIDE = inference_config['output_stride']
    INFERENCE_PRECISION = inference_config['precision']

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

MODELS_DIR = 'saved_models'
grayscale_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/grayscale'
rgb_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/rgb'

os.makedirs(grayscale_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)

data_module = CityscapesDataModule(dataset_config=dataset_config)

model = SegmentationModule()