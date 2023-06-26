import torch
from torch import nn
from torch.optim import Adam, SGD, LBFGS, Adadelta, Adamax, Adagrad, ASGD
from torch.optim.lr_scheduler import CyclicLR, PolynomialLR
from torch.optim.lr_scheduler import ExponentialLR, ConstantLR, StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

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


def get_optimizer(optimizer_config: dict, model: nn.Module):
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
    
    lr_warmup_method = lr_schedule_config.get('warmup_method', 'constant')
    lr_warmup_epochs = lr_schedule_config.get('warmup_epochs', 0)
    lr_warmup_decay = lr_schedule_config.get('warmup_decay', 0.1)
    
    main_schedule: str = lr_schedule_config.get('main_schedule')
    epochs = lr_schedule_config.get('main_schedule_epochs')
    main_schedule = main_schedule.lower()
    
    gamma = lr_schedule_config.get('gamma', 1.0)
    min_lr = lr_schedule_config.get('min_lr', 1e-3)
    max_lr = lr_schedule_config.get('max_lr', 1e-2)
    step_period = lr_schedule_config.get('step_period', 20)
    
    if main_schedule == 'polynomial':
        main_lr_schedule = PolynomialLR(
            optimizer=optimizer,
            total_iters=epochs, # iters_per_epoch * (epochs - lr_warmup_epochs),
            power=lr_schedule_config.get('power'),
        )
    elif main_schedule == "steplr":
        main_lr_schedule = StepLR(
            optimizer, step_size=step_period, gamma=gamma,
        )
    elif main_schedule == "cosineannealing":
        main_lr_schedule = CosineAnnealingLR(
            optimizer, 
            T_max = epochs - lr_warmup_epochs, 
            eta_min = min_lr,
        )
    elif main_schedule == "exponential":
        main_lr_schedule = ExponentialLR(
            optimizer, gamma=gamma,
        )
    elif main_schedule == 'cyclic':
        main_lr_schedule = CyclicLR(
            optimizer = optimizer,
            base_lr = min_lr,
            max_lr = max_lr,
            # step_size_up=
            # step_size_down=
            gamma = gamma,
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{main_schedule}'. Only StepLR, CosineAnnealingLR, "
            "ExponentialLR, PolynomialLR, CyclicLR are supported."
        )

    if lr_warmup_epochs > 0:
        if lr_warmup_method == "linear":
            warmup_lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif lr_warmup_method == "constant":
            warmup_lr_schedule = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. "
                "Only linear and constant are supported."
            )
        
        lr_schedule = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[warmup_lr_schedule, main_lr_schedule], 
                    milestones=[lr_warmup_epochs]
                )
    else:
        lr_schedule = main_lr_schedule
        
    return lr_schedule