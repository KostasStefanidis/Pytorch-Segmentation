import torch
import lightning.pytorch as pl
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

torch.set_float32_matmul_precision(INFERENCE_PRECISION)

checkpoint_dir = f'{LOGS_DIR}/saved_models/{MODEL_TYPE}/{MODEL_NAME}.ckpt'
grayscale_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/grayscale'
rgb_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/rgb'

os.makedirs(grayscale_path, exist_ok=True)
os.makedirs(rgb_path, exist_ok=True)

data_module = CityscapesDataModule(dataset_config=dataset_config)

model = SegmentationModule.load_from_checkpoint(checkpoint_dir)

trainer = pl.Trainer(
    accelerator='gpu',
    devices='0',
)

trainer.predict(model, datamodule=data_module, return_predictions=False)