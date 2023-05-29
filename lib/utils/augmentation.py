from typing import Any
import torchvision
import numpy as np
import torch
from torchvision.transforms.v2 import ColorJitter, RandomHorizontalFlip
from torchvision.transforms.v2 import GaussianBlur, Compose, RandomRotation, AugMix

torchvision.disable_beta_transforms_warning()

_ALL_SUPPORTED_AUGMENTATIONS = ['color_jitter', 'horizontal_flip', 'random_rotation', 'gaussian_blur']

_AUGMENTATION_NOT_SUPPORTED_ERROR = lambda aug: f'''\
{aug} is not in the list of supported transforms for augmentation. \
Supported augmentations are: {_ALL_SUPPORTED_AUGMENTATIONS}
'''

def get_augmentations(augmentation_config: dict):
    augmentation_list = []
    
    for augmentation in augmentation_config.keys():
        print('Augmentation: ', augmentation)
        if augmentation == 'color_jitter':
            brightness = augmentation_config.get('color_jitter').get('brightness', 0.0)
            contrast = augmentation_config.get('color_jitter').get('contrast', 0.0)
            saturation = augmentation_config.get('color_jitter').get('saturation', 0.0)
            hue = augmentation_config.get('color_jitter').get('hue', 0.0)
            augmentation_list.append(ColorJitter(brightness, contrast, saturation, hue))
            
        elif augmentation == 'horizontal_flip':
            flip_probability_value = augmentation_config.get('horizontal_flip')
            flip_probability = 0.5 if flip_probability_value is None else flip_probability_value
            augmentation_list.append(RandomHorizontalFlip(flip_probability))
            
        elif augmentation == 'random_rotation':
            degrees = augmentation_config.get('random_rotation', 10)
            augmentation_list.append(RandomRotation(degrees))
        
        elif augmentation == 'gaussian_blur':
            kernel_size = augmentation_config.get('gaussian_blur').get('kernel_size', 7)
            sigma = augmentation_config.get('gaussian_blur').get('sigma', (0.1, 3.0))
            augmentation_list.append(GaussianBlur(kernel_size, sigma))
            
        else:
            raise ValueError(_AUGMENTATION_NOT_SUPPORTED_ERROR(augmentation))
    
    augmentations = Compose(augmentation_list)
    print('')
    print(f'Using Augmentations: {augmentations}')
        
    return augmentations

class Mixup():
    def __init__(self, alpha) -> None:
        self.alpha = alpha
    
    def __call__(self, image, target, num_classes):
        
        indices = torch.randperm(image.size(0))
        image2 = image[indices]
        target2 = target[indices]
        lam = torch.FloatTensor([np.random.beta(self.alpha, self.alpha)])
        data = image * lam + image2 * (1 - lam)
        targets = target * lam + target2 * (1 - lam)
        return data, targets