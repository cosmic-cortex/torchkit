"""
Callback functions for training. The implementation details were
inspired by Keras, up to the point where I could have used those
classes straight away. However, I wanted to avoid depencency on
Keras, so I have reimplemented them.
"""


from collections import Container


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


class CallbackList(BaseCallback):
    def __init__(self, callbacks):
        assert isinstance(callbacks, Container), 'callbacks must be a Container'
        assert all([isinstance(callback, BaseCallback) for callback in callbacks]), 'every element of callbacks must' \
            'be an instance of a class inherited from BaseCallback'

        self.callbacks = callbacks

    def before_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.before_epoch()

    def after_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.after_epoch()
