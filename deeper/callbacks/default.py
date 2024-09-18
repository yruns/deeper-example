from logging import Logger

from deepspeed.runtime.engine import DeepSpeedEngine


class CallbackBase(object):
    trainer = None
    engine: DeepSpeedEngine
    logger: Logger = None

    def on_training_epoch_start(self):
        pass

    def on_training_epoch_end(self):
        pass

    def on_training_step_start(self):
        pass

    def on_training_step_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_training_phase_start(self):
        pass

    def on_training_phase_end(self):
        pass
