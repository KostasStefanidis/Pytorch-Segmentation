import torch
from torch import nn
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad, ASGD
from torch.optim.lr_scheduler import CyclicLR, PolynomialLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, StepLR, CosineAnnealingLR

def get_loss(loss: str):
        if loss in ['CrossEntropy', 'CrossEntropyLoss', 'crossentropy']:
            loss_fn = nn.CrossEntropyLoss()
        # elif loss in ['Dice, DiceLoss']:
        #     loss_fn = DiceLoss()
        # elif loss in ['Hybrid', 'HybridLoss']:
        #     loss_fn = HybridLoss()
        # elif loss in ['rmi', 'RMI', 'RmiLoss', 'RMILoss']:
        #     loss_fn = RMILoss()
        return loss_fn


def get_optimizer(optimizer_config:dict, model: nn.Module):

    optimizer_name = optimizer_config.get('name', 'Adam')
    lr = optimizer_config.get('learnin_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)
    momentum = optimizer_config.get('momentum', 0)
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    optimizer_dict = {
        'Adam' : Adam(trainable_params,
                        lr=lr,
                        weight_decay=weight_decay),
        'Adadelta' : Adadelta(trainable_params,
                                lr=lr,
                                weight_decay=weight_decay),
        'SGD' : SGD(trainable_params,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
    }
    
    return optimizer_dict[optimizer_name]


def get_lr_schedule(lr_schedule_config:dict, 
                    optimizer: torch.optim
                    ):
    
    if lr_schedule_config is None:
        return None

    schedule = lr_schedule_config.get('name')
    if schedule in ['Polynomial', 'PolynomialLr', 'PolynomialLR', 'polynomial']:
        decay_epochs = lr_schedule_config.get('decay_epochs')
        power = lr_schedule_config.get('power')
        lr_schedule = PolynomialLR(
            optimizer=optimizer,
            total_iters=decay_epochs, #*steps_per_epoch,
            power=power,
            verbose=True
        )
        
    elif schedule in ['CyclicLR', 'Cyclic', 'CyclicLr', 'cyclic']:
        lr_schedule = CyclicLR(
            optimizer = optimizer,
            base_lr = lr_schedule_config.get('min_lr', 1e-3),
            max_lr = lr_schedule_config.get('max_lr', 1e-2),
            # step_size_up=
            # step_size_down=
            gamma = lr_schedule_config.get('gamma', 1.0),
            verbose = True
        )

    return lr_schedule