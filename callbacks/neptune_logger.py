import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger:
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        trainer_params = {
            "n_epochs": self._config.getint('trainer', 'n_epochs'),
            "log_interval": self._config.getint('trainer', 'log_interval'),
            "seed": self._config.getint('trainer', 'seed'),
            "source": self._config.get('trainer', 'source')}

        data_loader_params = {
            "prioritized_sampler": self._config.getint('data_loader', 'prioritized_sampler'),
            "c_constant": self._config.getfloat('data_loader', 'c_constant'),
            "data_folder_name": self._config.get('data_loader', 'data_folder_name'),
            "data_file_name": self._config.get('data_loader', 'data_file_name'),
            "batch_size_train": self._config.getint('data_loader', 'batch_size_train'),
            "num_workers": self._config.getint('data_loader', 'num_workers'),
            "one_hot_encoding": self._config.getint('data_loader', 'one_hot_encoding'),
            "data_loader_type": self._config.get('data_loader', 'data_loader_type'),
            "cutmix_patch_min": self._config.getfloat('data_loader', 'cutmix_patch_min'),
            "cutmix_patch_max": self._config.getfloat('data_loader', 'cutmix_patch_max'),
            "cutmix_patch_num": self._config.getint('data_loader', 'cutmix_patch_num'),
            "exploit_type": self._config.get('data_loader', 'exploit_type')}

        augmentation_params = {
            "on": self._config.getint('augmentation', 'on'),
            "probs": self._config.getfloat('augmentation', 'probs'),
            "crop": self._config.getint('augmentation', 'crop'),
            "crop_probs": self._config.getfloat('augmentation', 'crop_probs'),
            "horizontal_flip": self._config.getint('augmentation', 'horizontal_flip'),
            "hf_probs": self._config.getfloat('augmentation', 'hf_probs'),
            "vertical_flip": self._config.getint('augmentation', 'vertical_flip'),
            "vf_probs": self._config.getfloat('augmentation', 'vf_probs'),
            "color_jitter": self._config.getint('augmentation', 'color_jitter'),
            "jitter_probs": self._config.getfloat('augmentation', 'jitter_probs'),
            "blur": self._config.getint('augmentation', 'blur'),
            "blur_probs": self._config.getfloat('augmentation', 'blur_probs'),
            "normalize": self._config.getint('augmentation', 'normalize')}

        model_params = {
            "model_type": self._config.get('model', 'model_type'),
            "pretrained": self._config.getint('model', 'pretrained')}

        callbacks_params = {
            "early_stopping_callback": self._config.getint('callbacks', 'early_stopping_callback'),
            "early_stopping_patience": self._config.getint('callbacks', 'early_stopping_patience')}

        agent_params = {
            "learning_rate": self._config.getfloat('agent', 'learning_rate'),
            "lr_decay": self._config.getint('agent', 'lr_decay'),
            "lr_decay_type": self._config.get('agent', 'lr_decay_type'),
            "lr_min": self._config.getfloat('agent', 'lr_min'),
            "warmup_epochs": self._config.getint('agent', 'warmup_epochs'),
            "warmup_lr_high": self._config.getfloat('agent', 'warmup_lr_high'),
            "momentum": self._config.getfloat('agent', 'momentum'),
            "weight_decay": self._config.getfloat('agent', 'weight_decay'),
            "loss": self._config.get('agent', 'loss'),
            "optimizer": self._config.get('agent', 'optimizer')}

        self.run["parameters/trainer_params"] = trainer_params
        self.run["parameters/data_loader_params"] = data_loader_params
        self.run["parameters/augmentation"] = augmentation_params
        self.run["parameters/model_params"] = model_params
        self.run["parameters/callbacks_params"] = callbacks_params
        self.run["parameters/optimizer_params"] = agent_params

        self.logger = NeptuneLogger(run=self.run, model=model)

    def save_checkpoint(self, checkpoint_name):
        self.logger.save_checkpoint(checkpoint_name=checkpoint_name)

    def save_model(self):
        self.logger.save_model()

    def stop(self):
        self.run.stop()
