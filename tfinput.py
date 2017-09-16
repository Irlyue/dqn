import tensorflow as tf


class TfInput(object):
    def __init__(self, name="untitled"):
        self.name = name

    def get(self):
        raise NotImplemented()

    def make_feed_dict(self, data):
        raise NotImplemented()


class PlaceholderInput(TfInput):
    def __init__(self, ph):
        super().__init__(ph.name)
        self._placeholder = ph

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class BatchInput(PlaceholderInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        super().__init__(tf.placeholder(dtype, shape=[None] + list(shape), name=name))
