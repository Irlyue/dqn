import gym
import simple
import numpy as np


from experience.model import q_func_cart_pole


def callback(local_vars, global_vars):
    """
    :param local_vars: dict
    :param global_vars: dict
    :return: bool
    """
    return local_vars['step'] > 100 and sum(local_vars['episode_rewards'][-101:-1]) > 100 * 50


def show_result():
    env = gym.make('CartPole-v0')
    act = simple.ActWrapper.load("cartpole_model.ckpt", num_cpus=1)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


def main():
    env = gym.make("CartPole-v0")
    act = simple.learn(env,
                       q_func_cart_pole,
                       exploration_fraction=0.35,
                       final_epsilon=0.1,
                       alpha=1e-3,
                       callback=callback)
    act.save("./cartpole_model.ckpt")


if __name__ == "__main__":
    show_result()
    # main()
