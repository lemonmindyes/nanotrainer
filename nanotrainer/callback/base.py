from abc import ABC


class Callback(ABC):
    # callback interface

    def on_train_begin(self, trainer):
        # Run once before entering the training loop
        pass

    def on_train_end(self, trainer):
        # Run once after exiting the training loop
        pass

    def on_epoch_begin(self, trainer):
        # Run once before each epoch
        pass

    def on_epoch_end(self, trainer):
        # Run once after each epoch
        pass

    def on_step_begin(self, trainer):
        # Run once before each step
        pass

    def on_step_end(self, trainer):
        # Run once after each step
        pass