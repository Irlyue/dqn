import tempfile
import dill
import os
import zipfile


import tf_util as U
import tensorflow as tf
import numpy as np


from .. import dqn
from replay_buffer import ReplayBuffer
from epsilon_schedule import LinearSchedule


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpus=1):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = dqn.build_act(**act_params)
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
          pre_run_steps=10000,
          exploration_fraction=0.1,
          final_epsilon=0.1):
    n_actions = env.action_space.n
    sess = U.make_session(num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)
    act, train, update_target, _ = dqn.build_train(
        make_obs_ph,
        q_func,
        n_actions,
        optimizer=tf.train.AdamOptimizer(alpha),
        gamma=gamma,
        param_noise=param_noise
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

    U.initialize()
    update_target()  # copy from the main network
    episode_rewards = []
    current_episode_reward = 0.0
    model_saved = False
    saved_mean_reward = None
    obs_t = env.reset()
    with tempfile.TemporaryDirectory() as td:
        model_file_path = os.path.join(td, "model")
        for step in range(n_steps):
            kwargs = {}
            if not param_noise:
                epsilon = exploration.value(step)
            else:
                assert False, "Not implemented"
            action = act(np.array(obs_t)[None], epsilon=epsilon, **kwargs)[0]
            obs_tp1, reward, done, _ = env.step(action)
            current_episode_reward += reward
            buffer.add(obs_t, action, reward, obs_tp1, done)
            if done:
                obs_t = env.reset()
                episode_rewards.append(current_episode_reward)
            # given sometime to fill in the buffer
            if step < pre_run_steps:
                continue
            if step % train_main_every == 0:
                obs_ts, actions, rewards, obs_tp1s, dones = buffer.sample(batch_size)
                weights = np.ones_like(dones)
                td_error = train(obs_ts, actions, rewards, obs_tp1s, dones, weights)
            if step % update_target_every == 0:
                update_target()
            mean_100eps_reward = float(np.mean(episode_rewards[-101:-1]))
            if done and print_every is not None and len(episode_rewards) % print_every == 0:
                print("step %d, episode %d, epsilon %.2f, current reward %.2f, running mean reward %.2f" %
                      (step, len(episode_rewards), epsilon, current_episode_reward, mean_100eps_reward))
            if checkpoint_every is not None and step % checkpoint_every:
                if saved_mean_reward is None or mean_100eps_reward > saved_mean_reward:
                    saved_mean_reward = mean_100eps_reward
                    U.save_state(model_file_path)
                    model_saved = True
                    if print_every is not None:
                        print("Dump model to file due to mean reward increase: %.2f -> %.2f" %
                              (saved_mean_reward, mean_100eps_reward))
        if model_saved:
            U.load_state(model_file_path)
            if print_every:
                print("Restore model from file with mean reward %.2f" % (saved_mean_reward,))
    return ActWrapper(act, act_params)
