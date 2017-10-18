import gym
import numpy as np
import simple


from experience.model import q_func_pong
from experience.env_wrappper import wrap_env


def callback(local_vars, global_vars):
    """
    :param local_vars: dict
    :param global_vars: dict
    :return: bool
    """
    # return True
    return local_vars['step'] > 1000 and sum(local_vars['episode_rewards'][-101:-1]) > 5.0


def show_result():
    env = wrap_env(gym.make("PongNoFrameskip-v4"))
    act = simple.ActWrapper.load("pong_model.ckpt", num_cpus=1)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


def main():
    env = wrap_env(gym.make("PongNoFrameskip-v4"))
    act = simple.learn(env,
                       q_func_pong,
                       n_steps=2000000,
                       exploration_fraction=0.20,
                       final_epsilon=0.01,
                       alpha=1e-3,
                       buffer_size=10000,
                       train_main_every=4,
                       update_target_every=1000,
                       gamma=0.99,
                       print_every=1,
                       pre_run_steps=10000,
                       callback=callback)
    # show_result(env, act)
    act.save("./pong_model.ckpt")


if __name__ == '__main__':
    show_result()
    # main()
