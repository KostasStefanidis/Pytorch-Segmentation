import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from torchsummary import summary
import numpy as np
import yaml
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
from lib.utils.callbacks import FreezeFeatureExtractor
import yaml
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Train segmentation network')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str, 
                        nargs='?')
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--seed', type=int, default=112)      
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    use_profiler = args.profiler
    use_early_stopping = args.early_stopping
    
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    with open(args.config, 'r') as config_file:
        config: dict = yaml.safe_load(config_file)

    logs_dir = config.get('logs_dir')
    model_config: dict = config.get('model_config')
    dataset_config: dict = config.get('dataset_config')
    train_config: dict = config.get('train_config')
    augmentation_config: dict = train_config.get('augmentations')
    distribute_config: dict = train_config.get('distribute')

    # Model Configuration
    MODEL_TYPE = model_config.get('architecture')
    MODEL_NAME = model_config.get('name')

    # --------------------------- Callbacks ----------------------------
    model_checkpoint_path = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
    model_checkpoint_callback = ModelCheckpoint(dirpath=logs_dir,
                                                filename=model_checkpoint_path,
                                                save_weights_only=False,
                                                monitor='val_Mean_IoU',
                                                mode='max',
                                                verbose=True)

    early_stopping_callback = EarlyStopping(patience=10,
                                            monitor='val_Mean_IoU',
                                            mode='max',
                                            min_delta=1e-6,
                                            verbose=True,
                                            strict=True,
                                            check_finite=True,
                                            log_rank_zero_only=True)
    
    finetune_backbone_callback = FreezeFeatureExtractor(unfreeze_at_epoch=20)

    callbacks = [
        model_checkpoint_callback, 
        RichProgressBar(),
        RichModelSummary(max_depth=2),
        LearningRateMonitor('epoch'),
        #finetune_backbone_callback
    ]

    if use_early_stopping:
        callbacks.append(early_stopping_callback)
    
    if train_config.get('swa') is not None:
        swa_config = train_config.get('swa')
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_config.get('lr', 1e-3),
            swa_epoch_start=swa_config.get('epoch_start', 0.8)
        )
        callbacks.append(swa_callback)

    logger = TensorBoardLogger(save_dir=f'{logs_dir}/Tensorboard_logs',
                               name=f'{MODEL_TYPE}',
                               version=f'{MODEL_NAME}',
                               default_hp_metric=False)

    if use_profiler:
        profiler = SimpleProfiler(dirpath=f'{logs_dir}/profiler_logs',
                                filename=f'{MODEL_TYPE}/{MODEL_NAME}')
    else:
        profiler = None

    torch.set_float32_matmul_precision(str(train_config.get('precision')))

    trainer = pl.Trainer(
        limit_train_batches=dataset_config.get('num_train_batches', 1.0),
        limit_val_batches=dataset_config.get('num_eval_batches', 1.0),
        max_epochs=train_config.get('epochs'),
        callbacks=callbacks,
        default_root_dir=logs_dir,
        logger=logger,
        accelerator='gpu',
        devices=distribute_config.get('devices'),
        strategy=distribute_config.get('strategy'),
        sync_batchnorm=distribute_config.get('sync_batchnorm'),
        profiler=profiler
    )
    
    model = SegmentationModule(
        model_config=model_config,
        train_config=train_config,
        logs_dir=logs_dir
    )

    datamodule = CityscapesDataModule(dataset_config, augmentation_config)
    
    trainer.fit(model, datamodule=datamodule)

    
if __name__ == '__main__':
    main()
