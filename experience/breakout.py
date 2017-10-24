import gym
import numpy as np
import simple


from experience.model import q_func_pong
from experience.env_wrappper import wrap_env, test_breakout_env


q_func_breakout = q_func_pong


def breakout_env_test():
    env = wrap_env(gym.make('Breakout-v0'))
    obs = env.reset()
    counter = 0
    while True:
        counter += 1
        env.render()
        obs, _, done, _ = env.step(np.random.randint(env.action_space.n))
        if done:
            print('counter = ', counter)
            break


def callback(local_vars, global_vars):
    """
    :param local_vars: dict
    :param global_vars: dict
    :return: bool
    """
    # return False
    return len(local_vars['episode_rewards']) > 100 and sum(local_vars['episode_rewards'][-101:-1]) > 15.0 * 100


def show_result():
    env = wrap_env(gym.make("BreakoutNoFrameskip-v4"))
    act = simple.ActWrapper.load("breakout_model.ckpt", num_cpus=1)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


def main():
    env = wrap_env(gym.make("BreakoutNoFrameskip-v4"))
    n_steps = 500000
    act = simple.learn(env,
                       q_func_breakout,
                       n_steps=n_steps,
                       exploration_fraction=0.2,
                       final_epsilon=0.01,
                       alpha=5e-4,
                       buffer_size=10000,
                       train_main_every=4,
                       update_target_every=1000,
                       gamma=0.99,
                       print_every=4,
                       pre_run_steps=10000,
                       callback=callback)
    # show_result(env, act)
    act.save("./breakout_model.ckpt")


if __name__ == '__main__':
    # test_breakout_env()
    # breakout_env_test()
    main()
    # show_result()

