from sl_prioritized_sampling.dataset.load_cifar100 import load_cifar100
from sl_prioritized_sampling.dataset.load_cifar10 import load_cifar10


def load_cifar(config, data_dir, num_classes=100):
    if num_classes == 100:
        cifar_data = load_cifar100(config, data_dir)
    elif num_classes == 10:
        cifar_data = load_cifar10(config, data_dir)
    else:
        raise ValueError

    return cifar_data
