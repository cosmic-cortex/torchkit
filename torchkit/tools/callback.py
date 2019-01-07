"""
Callback functions for training. The implementation details were
inspired by Keras, up to the point where I could have used those
classes straight away. However, I wanted to avoid depencency on
Keras, so I have reimplemented them.
"""


from collections import defaultdict


class BaseCallback:
    """
    Base class for callbacks.

    Args:
        model: torchkit model
    """
    def __init__(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass


class Logger(BaseCallback):
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def after_epoch(self, logs, *args, **kwargs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            if 'val_loss' not in logs.keys():
                print("Epoch no. %d \t training loss = %f" % (logs['epoch'], logs['train_loss']))
            else:
                print("Epoch no. %d \t training loss = %f \t validation loss = %f" %
                      (logs['epoch'], logs['train_loss'], logs['val_loss']))

    def get_logs(self):
        return self.logs


class CallbackList(BaseCallback):
    def __init__(self, callbacks):
        assert isinstance(callbacks, list), 'callbacks must be a list of callbacks'
        assert all([isinstance(callback, BaseCallback) for callback in callbacks]), 'every element of callbacks must' \
                                                                                    'be inherited from BaseCallback'

        self.callbacks = callbacks

    def before_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_epoch()

    def after_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_epoch()

    def append(self, item):
        assert isinstance(item, BaseCallback), 'item to be added must be inherited from BaseCallback'
        self.callbacks.append(item)

    def __add__(self, other):
        return CallbackList(self.callbacks + other.callbacks)
