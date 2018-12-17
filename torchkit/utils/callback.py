"""
Callback functions for training. The implementation details were
inspired by Keras, up to the point where I could have used those
classes straight away. However, I wanted to avoid depencency on
Keras, so I have reimplemented them.
"""


class BaseCallback:
    """
    Base class for callbacks.

    Args:
        model: torchkit model
    """
    def __init__(self, model):
        self.model = model


class CallbackList:
    def __init__(self, callbacks):
        assert isinstance(callbacks, list), 'callbacks must be a list'

        self.callbacks = callbacks
