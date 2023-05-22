import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
from lightning.pytorch.callbacks import ModelSummary, LearningRateFinder, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchsummary import summary
import yaml
from lib.datasets.cityscapes import CityscapesDataModule
from lib.models.base_module import SegmentationModule
import yaml
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Train segmentation network')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str, 
                        nargs='?')
    parser.add_argument('--seed', type=int, default=112)
    #parser.add_argument("--local_rank", type=int, default=-1)       
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    with open(args.config, 'r') as config_file:
        config: dict = yaml.safe_load(config_file)

    # TODO: Add default values if a variable is not defined in the config file
    LOGS_DIR = config.get('logs_dir')
    model_config: dict = config.get('model_config')
    dataset_config: dict = config.get('dataset_config')
    train_config: dict = config.get('train_config')
    augmentation_config: dict = train_config.get('augmentations')

    # Dataset Configuration
    DATASET = dataset_config.get('name')
    NUM_TRAIN_BATCHES = dataset_config.get('num_train_batches', 1.0)
    NUM_EVAL_BATCHES = dataset_config.get('num_eval_batches', 1.0)

    # Model Configuration
    MODEL_TYPE = model_config.get('architecture')
    MODEL_NAME = model_config.get('name')

    EPOCHS = train_config.get('epochs') #
    PRECISION = str(train_config.get('precision')) #
    DISTRIBUTE_STRATEGY = train_config.get('distribute').get('strategy')
    DEVICES = train_config.get('distribute').get('devices')
    USE_EARLY_STOPPING = train_config.get('early_stopping', False)
    
    # Stohastic weight averaging parameters
    SWA = train_config.get('swa')
    if SWA is not None:
        SWA_LRS = SWA.get('lr', 1e-3)
        SWA_EPOCH_START = SWA.get('epoch_start', 0.7)

    # --------------------------- Callbacks ----------------------------
    model_checkpoint_path = f'saved_models/{MODEL_TYPE}/{MODEL_NAME}'
    model_checkpoint_callback = ModelCheckpoint(dirpath=LOGS_DIR,
                                                filename=model_checkpoint_path,
                                                save_weights_only=False,
                                                monitor='val_Mean_IoU',
                                                mode='max',
                                                verbose=True)

    early_stopping_callback = EarlyStopping(patience=6,
                                            monitor='val_Mean_IoU',
                                            mode='max',
                                            min_delta=1e-6,
                                            verbose=True,
                                            strict=True,
                                            check_finite=True,
                                            log_rank_zero_only=True)

    callbacks = [model_checkpoint_callback, ModelSummary(max_depth=2)]

    if USE_EARLY_STOPPING:
        callbacks.append(early_stopping_callback)
    
    if SWA is not None:
        swa_callback = StochasticWeightAveraging(swa_lrs=SWA_LRS,
                                                 swa_epoch_start=SWA_EPOCH_START)
        callbacks.append(swa_callback)


    logger = TensorBoardLogger(save_dir=f'{LOGS_DIR}/Tensorboard_logs',
                               name=f'{MODEL_TYPE}',
                               version=f'{MODEL_NAME}')


    # --------------------------- Define Model -------------------------------
    model = SegmentationModule(
        model_config = model_config,
        train_config=train_config,
        logs_dir=LOGS_DIR
    )

    datamodule = CityscapesDataModule(dataset_config, augmentation_config)

    torch.set_float32_matmul_precision(PRECISION)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=DEVICES,
        limit_train_batches=NUM_TRAIN_BATCHES,
        limit_val_batches=NUM_EVAL_BATCHES,
        max_epochs=EPOCHS,
        callbacks=callbacks,
        default_root_dir=LOGS_DIR,
        logger=logger,
        #strategy=DISTRIBUTE_STRATEGY
        #profiler='simple',
        #sync_batchnorm=True,
    )

    trainer.fit(model, datamodule=datamodule)
    
if __name__ == '__main__':
    main()
