import torch
import lightning.pytorch as pl
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
from argparse import ArgumentParser
import yaml
import os
from lib.utils.callbacks import CityscapesPredictionWriter

def main():
    parser = ArgumentParser('')
    parser.add_argument('--config', type=str, nargs='?')
    parser.add_argument('--logs_dir', type=str, default='/mnt/logs')
    parser.add_argument('--dataset_config', type=str, default='config/dataset.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
    
    with open(args.dataset_config, 'r') as dataset_config_file:
        dataset_config: dict = yaml.safe_load(dataset_config_file).get('dataset_config')
    
    logs_dir = args.logs_dir
    model_config: dict = config.get('model_config')

    MODEL_TYPE = model_config.get('architecture')
    MODEL_NAME = model_config.get('name')

    checkpoint_dir = f'{logs_dir}/saved_models/{MODEL_TYPE}/{MODEL_NAME}.ckpt'
    grayscale_path = f'{logs_dir}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/grayscale'
    rgb_path = f'{logs_dir}/predictions/{MODEL_TYPE}/{MODEL_NAME}/test/rgb'

    os.makedirs(grayscale_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)

    torch.set_float32_matmul_precision('high')

    data_module = CityscapesDataModule(dataset_config=dataset_config)

    model = SegmentationModule.load_from_checkpoint(checkpoint_dir, logs_dir=logs_dir)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        logger=False,
        callbacks=CityscapesPredictionWriter(logs_dir, MODEL_TYPE, MODEL_NAME)
    )
    
    trainer.predict(model, datamodule=data_module, return_predictions=False)


if __name__ == '__main__':
    main()
