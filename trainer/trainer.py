import numpy as np
import random
import torch
import pickle
import copy


class Trainer(object):

    def __init__(self, config, criterion, optimizer, lr_scheduler, train_loader, val_loader, callbacks, model, sampler):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.model = model
        self.sampler = sampler

    @staticmethod
    def seed(seed_value: int):
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)

    def train(self):
        n_epochs = self.config.getint("trainer", "n_epochs")

        # Training & Validation
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(n_epochs):
            self.model.train()
            running_loss, correct_train = 0.0, 0

            for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
                inputs, labels, indices = inputs.to(device), labels.to(device), indices.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.clone())

                if bool(self.config.getint("callbacks", "neptune_logger_callback")):
                    neptune = self.callbacks["neptune_logger"]
                    neptune.run[neptune.logger.base_namespace]["metrics/train_loss"].append(loss.item())

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()

                if batch_idx % self.config.getint("trainer", "log_interval") == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(inputs), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))

            train_loss.append(running_loss / len(self.train_loader))
            last_train_acc = 100. * correct_train / len(self.train_loader.dataset)
            train_accuracy.append(last_train_acc)

            if bool(self.config.getint("callbacks", "neptune_logger_callback")):
                neptune = self.callbacks["neptune_logger"]
                neptune.run[neptune.logger.base_namespace]["metrics/train_acc"].append(
                    copy.deepcopy(last_train_acc / 100.))

            self.model.eval()
            running_loss, correct_val = 0.0, 0

            with torch.no_grad():
                for inputs, labels, _ in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.clone())

                    if bool(self.config.getint("callbacks", "neptune_logger_callback")):
                        neptune = self.callbacks["neptune_logger"]
                        neptune.run[neptune.logger.base_namespace]["metrics/val_loss"].append(loss.item())

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_val += (predicted == labels).sum().item()

            val_loss.append(running_loss / len(self.val_loader))
            last_val_acc = 100. * correct_val / len(self.val_loader.dataset)
            val_accuracy.append(last_val_acc)

            if bool(self.config.getint("callbacks", "neptune_logger_callback")):
                neptune = self.callbacks["neptune_logger"]
                neptune.run[neptune.logger.base_namespace]["metrics/val_acc"].append(copy.deepcopy(last_val_acc / 100.))

            print(
                f"Epoch {epoch + 1}/{n_epochs} - Train loss: {train_loss[-1]:.4f},"
                f"Train accuracy: {train_accuracy[-1]:.2f}%, Val loss: {val_loss[-1]:.4f},"
                f"Val accuracy: {val_accuracy[-1]:.2f}%")

            if bool(self.config.getint("callbacks", "early_stopping_callback")):
                self.callbacks["early_stopping"](val_loss[-1], self.model)
                if self.callbacks["early_stopping"].early_stop:
                    print("Early stopping")
                    break

            if bool(self.config.getint("agent", "lr_decay")) and bool(
                    self.config.getint("callbacks", "neptune_logger_callback")):
                neptune = self.callbacks["neptune_logger"]
                neptune.run[neptune.logger.base_namespace]["metrics/lr"].append(self.lr_scheduler.get_lr()[0])
                self.lr_scheduler.step()

        if bool(self.config.getint("callbacks", "early_stopping_callback")):
            self.model.load_state_dict(torch.load('checkpoint.pt'))
            file_name = self.config.get("model", "model_type") + "_weights.pth"
            torch.save(self.model.state_dict(), file_name)

        if bool(self.config.getint("callbacks", "neptune_logger_callback")):
            self.callbacks["neptune_logger"].save_model()
            self.callbacks["neptune_logger"].stop()

        history = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy}

        with open('training_history.pkl', 'wb') as file:
            pickle.dump(history, file)
