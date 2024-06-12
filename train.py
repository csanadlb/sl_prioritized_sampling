import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath("train.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configparser import ConfigParser

from sl_prioritized_sampling.factory.dataset_factory import DatasetFactory
from sl_prioritized_sampling.factory.data_loader_factory import DataLoaderFactory
from sl_prioritized_sampling.factory.model_factory import ModelFactory
from sl_prioritized_sampling.factory.agent_factory import AgentFactory
from sl_prioritized_sampling.factory.callbacks_factory import CallbacksFactory
from sl_prioritized_sampling.trainer.trainer import Trainer
from sl_prioritized_sampling.training_functions.custom_argument_parser import CustomArgParser


def main():
    my_config = ConfigParser()
    path = os.path.abspath('configuration.ini')
    my_config.read(path)

    arg_parser = CustomArgParser(config=my_config)
    parsed_config = arg_parser.get_parsed_config()

    Trainer.seed(parsed_config.getint('trainer', 'seed'))
    train_dataset, val_dataset = DatasetFactory.create(config=parsed_config)
    train_loader, val_loader, my_sampler = DataLoaderFactory.create(config=parsed_config, train_dataset=train_dataset,
                                                                    val_dataset=val_dataset)
    my_model = ModelFactory.create(config=parsed_config)
    my_callbacks = CallbacksFactory.create(config=parsed_config, model=my_model)
    my_criterion, my_optimizer, lr_scheduler = AgentFactory.create(config=parsed_config, model=my_model)

    trainer = Trainer(config=parsed_config,
                      criterion=my_criterion,
                      optimizer=my_optimizer,
                      lr_scheduler=lr_scheduler,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      callbacks=my_callbacks,
                      model=my_model,
                      sampler=my_sampler)

    trainer.train()


if __name__ == "__main__":
    main()
