import torch
import torch.optim as optim

from sl_prioritized_sampling.agent.lr_scheduler import WarmupCosineAnnealingLR, WarmupStepwiseLR
from sl_prioritized_sampling.factory.base_factory import BaseFactory
from sl_prioritized_sampling.agent.custom_loss import FocalLoss


class AgentFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        config = kwargs["config"]
        model = kwargs["model"]
        loss = config.get("agent", "loss")
        optimizer_name = config.get("agent", "optimizer")
        boundaries = [int(x.strip()) for x in config['agent']['boundaries'].split(',')]
        values = [float(x.strip()) for x in config['agent']['values'].split(',')]

        if loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif loss == "focal_loss":
            criterion = FocalLoss()
        else:
            raise NotImplementedError("Invalid loss type ('cross_entropy' or 'focal_loss')")

        if optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.getfloat("agent", "learning_rate"),
                                  momentum=config.getfloat("agent", "momentum"),
                                  weight_decay=config.getfloat("agent", "weight_decay"))
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.getfloat("agent", "learning_rate"),
                                   betas=(config.getfloat("agent", "beta1"), config.getfloat("agent", "beta2")),
                                   eps=config.getfloat("agent", "eps", fallback=1e-8),
                                   weight_decay=config.getfloat("agent", "weight_decay"),
                                   amsgrad=bool(config.getint("agent", "amsgrad", fallback=0)))
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=config.getfloat("agent", "learning_rate"),
                                      alpha=config.getfloat("agent", "alpha", fallback=0.99),
                                      eps=config.getfloat("agent", "eps", fallback=1e-8),
                                      weight_decay=config.getfloat("agent", "weight_decay"),
                                      momentum=config.getfloat("agent", "momentum", fallback=0),
                                      centered=bool(config.getint("agent", "centered", fallback=0)))
        else:
            raise NotImplementedError("Invalid optimizer type ('sgd' or 'adam' or 'rmsprop')")

        if bool(config.getint("agent", "lr_decay")):
            if config.get("agent", "lr_decay_type") == "cos":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                    T_max=config.getint("trainer", "n_epochs"))
            elif config.get("agent", "lr_decay_type") == "warmup_cos":
                T_max = config.getint("trainer", "n_epochs") - config.getint("agent", "warmup_epochs")
                lr_scheduler = WarmupCosineAnnealingLR(optimizer=optimizer,
                                                       T_max=T_max,
                                                       warmup_epochs=config.getint("agent", "warmup_epochs"),
                                                       eta_min=config.getfloat("agent", "lr_min"),
                                                       eta_max=config.getfloat("agent", "warmup_lr_high"))
            elif config.get("agent", "lr_decay_type") == "on_plateau":
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
            elif config.get("agent", "lr_decay_type") == "lin":
                lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer)
            elif config.get("agent", "lr_decay_type") == "exp":
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
            elif config.get("agent", "lr_decay_type") == "wstep":
                lr_scheduler = WarmupStepwiseLR(optimizer=optimizer,
                                                warmup_epochs=config.getint("agent", "warmup_epochs"),
                                                boundaries=boundaries,
                                                values=values)
            else:
                raise ValueError("Invalid learning rate scheduler "
                                 "('cos' or 'warmup_cos' or 'on_plateau' or 'lin' or 'exp')")
        else:
            lr_scheduler = None

        return criterion, optimizer, lr_scheduler
