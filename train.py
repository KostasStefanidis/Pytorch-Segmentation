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
    with open('config/Cityscapes.yaml', 'r') as config_file:
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
                           name=f'{MODEL_TYPE}/{MODEL_NAME}')


# --------------------------- Define Model -------------------------------
model = SegmentationModule(
    model_config = model_config,
    train_config=train_config,
    logs_dir=LOGS_DIR
)

data_module = CityscapesDataModule(dataset_config, augmentation_config)

torch.set_float32_matmul_precision(PRECISION)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=DEVICES,
    limit_train_batches=NUM_TRAIN_BATCHES,
    limit_val_batches=NUM_EVAL_BATCHES,
    max_epochs=EPOCHS,
    #precision=PRECISION,
    deterministic=False,
    callbacks=callbacks,
    default_root_dir=LOGS_DIR,
    logger=logger,
    #strategy=DISTRIBUTE_STRATEGY
    #profiler='simple',
    #sync_batchnorm=True,
)

trainer.fit(model, datamodule=data_module)