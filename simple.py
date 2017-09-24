import tempfile
import dill
import os
import zipfile


import tf_util as U
import tensorflow as tf
import numpy as np


from replay_buffer import ReplayBuffer
from epsilon_schedule import LinearSchedule
from build_graph import build_act, build_train


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpus=1):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = build_act(**act_params)
        sess = U.make_session(num_cpus=num_cpus)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "packed.zip")
            with open(filepath, "wb") as f:
                f.write(model_data)
            zipfile.ZipFile(filepath, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))
        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            filepath = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(filepath, "w") as zipf:
                for root, dirs, filenames in os.walk(filepath):
                    for filename in filenames:
                        fully_path = os.path.join(root, filename)
                        # leave the packed.zip file alone
                        if fully_path != filepath:
                            zipf.write(fully_path, os.path.relpath(fully_path, td))
            with open(filepath, "rb") as f:
                model_data = f.read()
            with open(path, "wb") as f:
                dill.dump((model_data, self._act_params), f)


def learn(env,
          q_func,
          alpha=1e-5,
          num_cpu=1,
          n_steps=100000,
          update_target_every=500,
          train_main_every=1,
          print_every=50,
          checkpoint_every=10000,
          buffer_size=50000,
          gamma=1.0,
          batch_size=32,
          param_noise=False,
          pre_run_steps=1000,
          exploration_fraction=0.1,
          final_epsilon=0.1,
          callback=None):
    """
    :param env: gym.Env, environment from OpenAI
    :param q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the q function takes the following inputs:
        input_ph: tf.placeholder, network input
        n_actions: int, number of possible actions
        scope: str, specifying the variable scope
        reuse: bool, whether to reuse the variable given in `scope`
    :param alpha: learning rate
    :param num_cpu: number of cpu to use
    :param n_steps: number of training steps
    :param update_target_every: frequency to update the target network
    :param train_main_every: frequency to update(train) the main network
    :param print_every: how often to print message to console
    :param checkpoint_every: how often to save the model.
    :param buffer_size: size of the replay buffer
    :param gamma: int, discount factor
    :param batch_size: int, size of the input batch
    :param param_noise: bool, whether to use parameter noise
    :param pre_run_steps: bool, pre-run steps to fill in the replay buffer. And only
        after `pre_run_steps` steps, will the main and target network begin to update.
    :param exploration_fraction: float, between 0 and 1. Fraction of the `n_steps` to
        linearly decrease the epsilon. After that, the epsilon will remain unchanged.
    :param final_epsilon: float, final epsilon value, usually a very small number
        towards zero.
    :param callback: (dict, dict) -> bool
        a function to decide whether it's time to stop training, takes following inputs:
        local_vars: dict, the local variables in the current scope
        global_vars: dict, the global variables in the current scope
    :return: ActWrapper, a callable function
    """
    n_actions = env.action_space.n
    sess = U.make_session(num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)
    act, train, update_target, debug = build_train(
        make_obs_ph,
        q_func,
        n_actions,
        optimizer=tf.train.AdamOptimizer(alpha),
        gamma=gamma,
        param_noise=param_noise,
        grad_norm_clipping=10
    )
    act_params = {
        "q_func": q_func,
        "n_actions": n_actions,
        "make_obs_ph": make_obs_ph
    }
    buffer = ReplayBuffer(buffer_size)
    exploration = LinearSchedule(schedule_steps=int(exploration_fraction * n_steps),
                                 final_p=final_epsilon,
                                 initial_p=1.0)
    writer = tf.summary.FileWriter("./log", sess.graph)

    U.initialize()
    writer.close()
    update_target()  # copy from the main network
    episode_rewards = []
    current_episode_reward = 0.0
    model_saved = False
    saved_mean_reward = 0.0
    obs_t = env.reset()
    with tempfile.TemporaryDirectory() as td:
        model_file_path = os.path.join(td, "model")
        for step in range(n_steps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            kwargs = {}
            if not param_noise:
                epsilon = exploration.value(step)
            else:
                assert False, "Not implemented"
            action = act(np.array(obs_t)[None], epsilon=epsilon, **kwargs)[0]
            obs_tp1, reward, done, _ = env.step(action)
            current_episode_reward += reward
            buffer.add(obs_t, action, reward, obs_tp1, done)
            obs_t = obs_tp1
            if done:
                obs_t = env.reset()
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
            # given sometime to fill in the buffer
            if step < pre_run_steps:
                continue
            # q_value = debug["q_values"]
            # if step % 1000 == 0:
            #     print(q_value(np.array(obs_t)[None]))
            if step % train_main_every == 0:
                obs_ts, actions, rewards, obs_tp1s, dones = buffer.sample(batch_size)
                rewards = (rewards - np.mean(rewards)) / np.max(rewards)
                weights = np.ones_like(dones)
                td_error = train(obs_ts, actions, rewards, obs_tp1s, dones, weights)
            if step % update_target_every == 0:
                update_target()
            mean_100eps_reward = float(np.mean(episode_rewards[-101:-1]))
            if done and print_every is not None and len(episode_rewards) % print_every == 0:
                print("step %d, episode %d, epsilon %.2f, running mean reward %.2f" %
                      (step, len(episode_rewards), epsilon, mean_100eps_reward))
            if checkpoint_every is not None and step % checkpoint_every == 0:
                if saved_mean_reward is None or mean_100eps_reward > saved_mean_reward:
                    U.save_state(model_file_path)
                    model_saved = True
                    if print_every is not None:
                        print("Dump model to file due to mean reward increase: %.2f -> %.2f" %
                              (saved_mean_reward, mean_100eps_reward))
                    saved_mean_reward = mean_100eps_reward
        if model_saved:
            U.load_state(model_file_path)
            if print_every:
                print("Restore model from file with mean reward %.2f" % (saved_mean_reward,))
    return ActWrapper(act, act_params)
