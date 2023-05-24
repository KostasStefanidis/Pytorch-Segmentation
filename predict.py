import torch
import lightning.pytorch as pl
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
from argparse import ArgumentParser
import yaml
import os

def main():
    parser = ArgumentParser('')
    parser.add_argument('--config', type=str, nargs='?')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

        LOGS_DIR = config.get('logs_dir')

        dataset_config = config.get('dataset_config')
        model_config = config.get('model_config')
        inference_config = config.get('inference_config')

        MODEL_TYPE = model_config.get('architecture')
        MODEL_NAME = model_config.get('name')

        INFERENCE_OUTPUT_STRIDE = inference_config.get('output_stride', 16)
        INFERENCE_PRECISION = inference_config.get('precision')

    torch.set_float32_matmul_precision(INFERENCE_PRECISION)

    checkpoint_dir = f'{LOGS_DIR}/saved_models/{MODEL_TYPE}/{MODEL_NAME}.ckpt'
    grayscale_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/grayscale'
    rgb_path = f'{LOGS_DIR}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/rgb'

    os.makedirs(grayscale_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)

    data_module = CityscapesDataModule(dataset_config=dataset_config)

    model = SegmentationModule.load_from_checkpoint(checkpoint_dir, logs_dir=LOGS_DIR)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=False,
    )

    trainer.predict(model, datamodule=data_module, return_predictions=False)
    
if __name__ == '__main__':
    main()
