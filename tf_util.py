import tensorflow as tf
import os
import collections


from tfinput import *
from functions import Function


def make_session(num_cpus):
    """Currently just return a normal session and ignore the parameter"""
    # TODO
    return tf.Session()


INITIALIZED_VARIABLES = set()


def initialize():
    unintialized = set(tf.global_variables()) - INITIALIZED_VARIABLES
    get_latest_session().run(tf.variables_initializer(unintialized))
    INITIALIZED_VARIABLES.update(unintialized)


def load_state(filename):
    saver = tf.train.Saver()
    saver.restore(get_latest_session(), filename)


def save_state(filename):
    create_if_not_exists(os.path.dirname(filename))
    saver = tf.train.Saver()
    saver.save(get_latest_session(), filename)


def create_if_not_exists(dirname):
    """Create the specified directory if not exists"""
    os.makedirs(dirname, exist_ok=True)


def get_latest_session():
    """Return the latest created default session"""
    return tf.get_default_session()


def ensure_tf_input(ph):
    if isinstance(ph, TfInput):
        return ph
    elif is_placeholder(ph):
        return PlaceholderInput(ph)
    else:
        raise ValueError("TfInput and tf.placeholder only")


def is_placeholder(thing):
    return type(thing) is tf.Tensor and len(thing.op.inputs) == 0


def get_variable_name(input_):
    """
    :param input_: a placeholder or variable, typically has name format
    like deepq/action:0
    :return: str, given a name format like deepq/action:0, action will be returned
    """
    input_name = input_.name.split(":")[0]
    input_name = input_name.split("/")[-1]
    return input_name


def make_function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        raise NotImplemented()
    else:
        f = Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


def huber_loss(x, delta=1.0):
    with tf.variable_scope("huber_loss", reuse=False):
        delta = tf.constant(delta, name="delta")
        flag = tf.abs(x) < delta
        return tf.where(flag, 0.5 * tf.square(x), delta * (tf.abs(x) - 0.5 * delta))


def absolute_scope_name(scope):
    return scope_name() + "/" + scope


def scope_name():
    return tf.get_variable_scope().name


def scope_vars(scope, trainable_only=False):
    """
    :param scope: str or tensorflow scope variable
    :param trainable_only: if true will only return those variables flagged as
    trainable
    :return:
    """
    return tf.get_collection(
        tf.GraphKeys.TABLE_INITIALIZERS if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    with tf.variable_scope("gradient_clipping", reuse=False):
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)
