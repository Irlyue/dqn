import tensorflow as tf
import numpy as np
import tf_util as U


from tfinput import TfInput


class Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for input_ in inputs:
            if not issubclass(type(input_), TfInput):
                assert len(input_.op.inputs) == 0, "Input should all be placeholders"
        self.inputs = inputs
        updates = updates or []
        self.update = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update]
        self.givens = givens if givens is not None else {}
        self.check_nan = check_nan

    @staticmethod
    def feed_input(feed_dict, input_, value):
        if issubclass(type(input_), TfInput):
            feed_dict.update(input_.make_feed_dict(value))
        elif U.is_placeholder(input_):
            feed_dict[input_] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        for input_, value in zip(self.inputs, args):
            self.feed_input(feed_dict, input_, value)
        # add in the kwargs
        processed_kwarg_names = set()
        for input_ in self.inputs[len(args):]:
            input_name = U.get_variable_name(input_)
            assert input_name not in processed_kwarg_names, "Got duplicated keyword argument"
            if input_name in kwargs:
                processed_kwarg_names.add(input_name)
                self.feed_input(feed_dict, input_, kwargs.pop(input_name))
            else:
                assert input_ in self.givens, "Missing argument %s" % (input_name,)
        assert len(kwargs) == 0, "Got extra keyword argument"
        # add in given arguments, won't update those argument already in feed_dict
        for input_ in self.givens:
            feed_dict[input_] = feed_dict.get(input_, self.givens[input_])
        # run it
        results = U.get_latest_session().run(self.outputs_update, feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(item) for item in results):
                raise RuntimeError("Nan detected!")
        return results
