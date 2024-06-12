from sl_prioritized_sampling.factory.base_factory import BaseFactory
from sl_prioritized_sampling.callbacks.torch_earlystopping import EarlyStopping
from sl_prioritized_sampling.callbacks.neptune_logger import CustomNeptuneLogger


class CallbacksFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        my_callbacks = {}
        config = kwargs["config"]

        if bool(config.getint("callbacks", "early_stopping_callback")):
            patience = kwargs["config"].getint("callbacks", "early_stopping_patience")
            early_stopping_callback = EarlyStopping(patience=patience, verbose=True)
            my_callbacks["early_stopping"] = early_stopping_callback

        if bool(config.getint("callbacks", "neptune_logger_callback")):
            neptune_logger_callback = CustomNeptuneLogger(token=config.get("callbacks", "neptune_token"),
                                                          project=config.get("callbacks", "neptune_project"),
                                                          config=config)
            neptune_logger_callback.start_logging(kwargs["model"])
            my_callbacks["neptune_logger"] = neptune_logger_callback

        return my_callbacks
