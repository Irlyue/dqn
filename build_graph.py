import tensorflow as tf
import tf_util as U


def get_dimension(input_, axis):
    shape = tf.shape(input_)
    return shape[axis]


def build_act(make_obs_ph, q_func, n_actions, scope="deepq", reuse=None):
    """
    :param make_obs_ph:
    :param q_func:
    :param n_actions: number of actions
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # placeholders
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, shape=[], name="stochastic")
        epsilon_ph = tf.placeholder(tf.float32, shape=[], name="epsilon")

        epsilon = tf.get_variable("epsilon", shape=(), initializer=tf.constant_initializer(0))
        values = q_func(observations_ph.get(), n_actions, scope="q_func")
        deterministic_actions = tf.argmax(values, axis=1)
        batch_size = tf.stack([get_dimension(observations_ph, 0)])
        random_actions = sample_from_uniform(batch_size, n_actions, tf.int64)
        choose_random_flag = sample_from_uniform(batch_size, 1, tf.float32) < epsilon
        final_actions = tf.where(choose_random_flag, random_actions, deterministic_actions)
        output_actions = tf.cond(stochastic_ph, lambda: final_actions, lambda: deterministic_actions)
        update_epsilon_op = epsilon.assign(tf.cond(epsilon >= 0, lambda: epsilon_ph, lambda: epsilon))
        act = U.make_function(inputs=(observations_ph, stochastic_ph, epsilon_ph),
                              outputs=output_actions,
                              givens={epsilon_ph: -1.0, stochastic_ph: True},
                              updates=[update_epsilon_op])
        return act


def sample_from_uniform(size, maxval, dtype):
    return tf.random_uniform(size, minval=0, maxval=maxval, dtype=dtype)


def build_train(make_obs_ph, q_func, n_actions, optimizer, grad_norm_clipping=None,
                gamma=1.0, double_q=True, scope="deepq", reuse=None, param_noise=False,
                param_noise_filter_func=None):
    """
    :param make_obs_ph: str -> tf.placeholder
    a function that create a placeholder given that name
    :param q_func: input, n_actions, scope, reuse -> tf.Tensor
    the model that takes the following paramters:
        input: tf.placeholder
        n_actions: int, number of actions
        scope: str
        reuse: bool, whether to reuse the variables from the scope
    :param n_actions: number of actions
    :param optimizer:
    :param grad_norm_clipping:
    :param gamma:
    :param double_q: bool, whether to use double q value or not
    :param scope:
    :param reuse:
    :param param_noise:
    :param param_noise_filter_func:
    :return: a bunch of functions
        act_f: function to generate actions
        train_f: function to update the main network
        update_target_f: function used to update the target network
        {}: other useful functions
    """
    if param_noise:
        raise NotImplemented()
    else:
        act_f = build_act(make_obs_ph, q_func, n_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        obs_t_ph = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int64, shape=[None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, shape=[None], name="reward")
        obs_tp1_ph = tf.placeholder(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, shape=[None], name="done")
        weights_ph = tf.placeholder(tf.float32, shape=[None], name="weight")
        # q values
        q_t = q_func(obs_t_ph.get(), n_actions, scope="q_func", reuse=True)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        # target q values
        q_target_tp1 = q_func(obs_tp1_ph.get(), n_actions, scope="q_target_func")
        q_target_vars = U.scope_vars(U.absolute_scope_name("q_target_func"))
        if double_q:
            q_tpl1 = q_func(obs_tp1_ph.get(), n_actions, scope='q_func', reuse=True)
            responsible_actions = tf.argmax(q_tpl1, axis=1)
            double_q_value = tf.reduce_sum(q_target_tp1 * tf.one_hot(responsible_actions, n_actions), axis=1)
        else:
            raise NotImplemented()
        double_q_value_masked = (1.0 - done_mask_ph) * double_q_value
        q_true_value = rew_t_ph + gamma * double_q_value_masked
        q_current_value = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, n_actions), axis=1)
        td_error = q_current_value - tf.stop_gradient(q_true_value)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(errors * weights_ph)
        if grad_norm_clipping is not None:
            raise NotImplemented()
        else:
            train_op = optimizer.minimize(weighted_error, var_list=q_func_vars)
        update_target_ops = []
        for qvar, qtarget_var in zip(sorted(q_func_vars, key=lambda v: v.name),
                                     sorted(q_target_vars, key=lambda v: v.name)):
            update_target_ops.append(qtarget_var.assign(qvar))
        update_target_network = tf.group(*update_target_ops)
        # create callable function
        train_f = U.make_function(
            inputs=[obs_t_ph, act_t_ph, rew_t_ph, obs_tp1_ph, done_mask_ph, weights_ph],
            outputs=td_error,
            updates=[train_op]
        )
        update_target_f = U.make_function([], [], updates=[update_target_network])
        q_values_f = U.make_function([obs_t_ph], q_t)
        return act_f, train_f, update_target_f, {'q_values': q_values_f}
