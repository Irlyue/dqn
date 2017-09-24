import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


import simple


def q_func(input_ph, n_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        hidden_layer = layers.fully_connected(inputs=input_ph,
                                              num_outputs=64,
                                              activation_fn=tf.nn.relu)
        outputs = layers.fully_connected(inputs=hidden_layer,
                                         num_outputs=n_actions,
                                         activation_fn=None)
        return outputs


def callback(local_vars, global_vars):
    """
    :param local_vars: dict
    :param global_vars: dict
    :return: bool
    """
    return local_vars['step'] > 100 and sum(local_vars['episode_rewards'][-101:-1]) > 100 * 185


def show_result(env, act_f):
    counter = 0
    obs_t = env.reset()
    done = False
    while counter < 100000:
        counter += 1
        env.render()
        step = act_f(np.array(obs_t)[None], epsilon=0.0)[0]
        obs_tp1, reward, done, info = env.step(step)
        obs_t = obs_tp1
        if done:
            obs_t = env.reset()


def main():
    tf.reset_default_graph()
    env = gym.make("CartPole-v0")
    act = simple.learn(env,
                       q_func,
                       exploration_fraction=0.35,
                       final_epsilon=0.1,
                       alpha=5e-4,
                       callback=callback)
    show_result(env, act)


if __name__ == "__main__":
    main()
