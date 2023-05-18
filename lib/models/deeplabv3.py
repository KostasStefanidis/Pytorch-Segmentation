import torch
from torch import Tensor
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn import Conv2d, ConvTranspose2d, Dropout, FeatureAlphaDropout
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import Sequential, Module, ModuleList, ModuleDict
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1
from torchvision.models.efficientnet import efficientnet_b2, efficientnet_b3
from torchvision.models.efficientnet import efficientnet_b4, efficientnet_b5
from torchvision.models.efficientnet import efficientnet_b6, efficientnet_b7
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.regnet import regnet_x_8gf, regnet_x_16gf, regnet_x_32gf
from torchvision.models.regnet import regnet_y_8gf, regnet_y_16gf, regnet_y_32gf
from torchvision.models import RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights
from torchvision.models import RegNet_Y_8GF_Weights, EfficientNet_V2_M_Weights
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, ResNet101_Weights
from lib.models.base_module import SegmentationModule
# MobileNet_V3_Large_Weights.IMAGENET1K_V1            # 74.042 | 91.34  |  5.5M  | 0.22GF
# MobileNet_V3_Large_Weights.IMAGENET1K_V2            # 75.274 | 92.566 |  5.5M  | 0.22GF
# ResNet50_Weights.IMAGENET1K_V2                      # 80.858 | 95.434 | 25.6M  | 4.09GF
# ResNet101_Weights.IMAGENET1K_V2                     # 81.886 | 95.78  | 44.5M  | 7.8GF
# RegNet_Y_8GF_Weights.IMAGENET1K_V1                  # 80.032 | 95.048 | 39.4M  | 8.47GF
# RegNet_Y_8GF_Weights.IMAGENET1K_V2                  # 82.828 | 96.33  | 39.4M  | 8.47GF
# RegNet_Y_16GF_Weights.IMAGENET1K_V1                 # 80.424 | 95.24  | 83.6M  | 15.91GF
# RegNet_Y_16GF_Weights.IMAGENET1K_V2                 # 82.886 | 96.328 | 83.6M  | 15.91GF
# RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1     # 83.976 | 97.244 | 83.6M  | 15.91GF
# RegNet_Y_32GF_Weights.IMAGENET1K_V1                 # 80.878 | 95.34  | 145.0M | 32.28GF
# RegNet_Y_32GF_Weights.IMAGENET1K_V2                 # 83.368 | 96.498 | 145.0M | 32.28GF
# RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1     # 
# EfficientNet_V2_M_Weights.IMAGENET1K_V1             # 85.112 | 97.156 | 54.1M  | 24.58GF


from torchvision.models.segmentation.deeplabv3 import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


class DeepLabV3(SegmentationModule):
    def __init__(self, 
                 model: Module, 
                 num_classes: int, 
                 train_config: dict = None, 
                 prediction_dir: str = None, 
                 model_type: str = None, 
                 model_name: str = None
                 ) -> None:
        
        super().__init__(model, num_classes, train_config, prediction_dir, model_type, model_name)
    
    

def get_backbone(backbone_name: str, 
                 input_tensor: Tensor, 
                 freeze_backbone: bool, 
                 unfreeze_at: str, 
                 output_stride: int = None, 
                 depth: int = None):    
    if output_stride is None:
        output_stride = 32
    
    if output_stride != 32 and 'EfficientNetV2' not in backbone_name:
        raise NotImplementedError(f'output_stride other than 32 is not implemented for backbone {backbone_name}. \
                                    To specify a different value for output_stride use EfficientNetV2 as network backbone.')
    
    backbone_func = eval(backbone_name)
    
    if 'EfficientNetV2' in backbone_name:
        backbone_ = backbone_func(output_stride=output_stride,
                                  include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    else:
        backbone_ = backbone_func(include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    
    layer_names = BACKBONE_LAYERS[backbone_name]
    if depth is None:
        depth = len(layer_names)

    X_skip = []
    # get the output of intermediate backbone layers to use them as skip connections
    for i in range(depth):
        X_skip.append(backbone_.get_layer(layer_names[i]).output)
        
    # backbone = Model(inputs=input_tensor, outputs=X_skip, name=f'{backbone_name}_backbone')
    
    # if freeze_backbone:
    #     backbone.trainable = False
    # elif unfreeze_at is not None:
    #     layer_dict = {layer.name: i for i,layer in enumerate(backbone.layers)}
    #     unfreeze_index = layer_dict[unfreeze_at]
    #     for layer in backbone.layers[:unfreeze_index]:
    #         layer.trainable = False
    # return backbone


BACKBONE_LAYERS = {
        'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
        'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
        'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
        'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
        'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
        'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
        # MobileNets
        'MobileNet' : ('conv_pw_1_relu', 'conv_pw_3_relu', 'conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu'),
        'MobileNetV2' : ('block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'),
        'MobileNetV3Small' : ('multiply', 're_lu_3', 'multiply_1', 'multiply_11', 'multiply_17'),
        'MobileNetV3Large' : ('re_lu_2', 're_lu_6', 'multiply_1', 'multiply_13', 'multiply_19'),
        # EfficientNet
        'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        # EfficientNetV2
        'EfficientNetV2S' : ('block1b_add', 'block2d_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2M' : ('block1c_add', 'block2e_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2L' : ('block1d_add', 'block2g_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
        # RegNetX
        'RegNetX002' : (
            'regnetx002_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx002_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx002_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx002_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx002_Stage_3_XBlock_4_exit_relu'
            ),
        'RegNetX004' : (
            'regnetx004_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx004_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx004_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx004_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx004_Stage_3_XBlock_4_exit_relu'
            ),
        'RegNetX006' : (
            'regnetx006_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx006_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx006_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx006_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx006_Stage_3_XBlock_4_exit_relu'
            ),
        'RegNetX008' : (
            'regnetx008_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx008_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx008_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx008_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx008_Stage_3_XBlock_4_exit_relu'
            ),
        'RegNetX016' : (
            'regnetx016_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx016_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx016_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx016_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx016_Stage_3_XBlock_1_exit_relu'
            ),
        'RegNetX032' : (
            'regnetx032_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx032_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx032_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx032_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx032_Stage_3_XBlock_1_exit_relu'
            ),
        'RegNetX040' : (
            'regnetx040_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx040_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx040_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx040_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx040_Stage_3_XBlock_1_exit_relu'
            ),
        'RegNetX064' : (
            'regnetx064_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx064_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx064_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx064_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx064_Stage_3_XBlock_0_exit_relu'
            ),
        'RegNetX080' : (
            'regnetx080_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx080_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx080_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx080_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx080_Stage_3_XBlock_0_exit_relu'
            ),
        'RegNetX120' : (
            'regnetx120_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx120_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx120_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx120_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx120_Stage_3_XBlock_0_exit_relu'
            ),
        'RegNetX160' : (
            'regnetx160_Stage_0_XBlock_0_conv_1x1_1_relu', 
            'regnetx160_Stage_1_XBlock_0_conv_1x1_1_relu', 
            'regnetx160_Stage_2_XBlock_0_conv_1x1_1_relu', 
            'regnetx160_Stage_3_XBlock_0_conv_1x1_1_relu', 
            'regnetx160_Stage_3_XBlock_0_exit_relu'
            ),
        'RegNetX320' : ('regnetx320_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_3_XBlock_0_exit_relu'),
        # RegNetY
        'RegNetY002' : ('regnety002_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_3_YBlock_4_exit_relu'),
        'RegNetY004' : ('regnety004_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_3_YBlock_4_exit_relu'),
        'RegNetY006' : ('regnety006_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_3_YBlock_3_exit_relu'),
        'RegNetY008' : ('regnety008_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_3_YBlock_1_exit_relu'),
        'RegNetY016' : ('regnety016_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_3_YBlock_1_exit_relu'),
        'RegNetY032' : ('regnety032_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_3_YBlock_0_exit_relu'),
        'RegNetY040' : ('regnety040_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_3_YBlock_0_exit_relu'),
        'RegNetY064' : ('regnety064_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_3_YBlock_0_exit_relu'),
        'RegNetY080' : ('regnety080_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_3_YBlock_0_exit_relu'),
        'RegNetY120' : ('regnety120_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_3_YBlock_0_exit_relu'),
        'RegNetY160' : ('regnety160_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_3_YBlock_0_exit_relu'),
        'RegNetY320' : ('regnety320_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_3_YBlock_0_exit_relu'),
    }