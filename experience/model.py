import tensorflow  as tf


from tensorflow.contrib import layers


def q_func_pong(input_, n_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        with tf.variable_scope('convnet'):
            out = input_
            for n_filters, filter_dim, stride in convs:
                out = layers.conv2d(inputs=out,
                                    num_outputs=n_filters,
                                    kernel_size=filter_dim,
                                    stride=stride,
                                    activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        hiddens = [256]
        with tf.variable_scope('mlp'):
            for hidden_dim in hiddens:
                out = layers.fully_connected(inputs=out,
                                             num_outputs=hidden_dim,
                                             activation_fn=tf.nn.relu)
        out = layers.fully_connected(inputs=out,
                                     num_outputs=n_actions,
                                     activation_fn=None)
        return out


def q_func_cart_pole(input_ph, n_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hidden_layer = layers.fully_connected(inputs=input_ph,
                                              num_outputs=64,
                                              activation_fn=tf.nn.relu)
        outputs = layers.fully_connected(inputs=hidden_layer,
                                         num_outputs=n_actions,
                                         activation_fn=None)
        return outputs