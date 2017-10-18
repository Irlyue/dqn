import gym
import numpy as np
import matplotlib.pyplot as plt


from collections import deque
from scipy.misc import imresize
from gym import spaces


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self.obs_buffer = deque(maxlen=2)
        self.skip_frames = skip

    def _step(self, action):
        total_reward = 0.0
        for step in range(self.skip_frames):
            obs_tp1, reward, done, info = self.env.step(action)
            self.obs_buffer.append(obs_tp1)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        self.obs_buffer.clear()
        self.obs_buffer.append(self.env.reset())
        return self.obs_buffer[0]


class ProcessImageEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessImageEnv, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(56, 46, 1))

    def _observation(self, obs):
        return ProcessImageEnv.process(obs)

    @staticmethod
    def process(img):
        gray_img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        UP = 30
        BOTTOM = 200
        LEFT = 10
        RIGHT = 150
        target_img = gray_img[UP:BOTTOM, LEFT:RIGHT]
        shp = target_img.shape
        # downsample by factor of 2
        resized_img = imresize(target_img, (shp[0] // 3, shp[1] // 3))
        return resized_img[:, :, None]


class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.stack_frames = k
        self.frames = deque(maxlen=k)
        shp = self.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def _reset(self):
        obs = self.env.reset()
        for step in range(self.stack_frames):
            self.frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)

    def _step(self, action):
        obs_tp1, reward, done, info = self.env.step(action)
        self.frames.append(obs_tp1)
        return self._get_obs(), reward, done, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def wrap_env(env):
    env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env)
    env = ProcessImageEnv(env)
    env = FrameStackEnv(env, 4)
    env = ScaledFloatFrame(env)
    return env


def test_pong():
    """Simple test"""
    env = gym.make('PongNoFrameskip-v4')
    env = wrap_env(env)
    obs = env.reset()
    for i in range(50):
        obs, _, done, _ = env.step(2)
        env.render()
        if done:
            obs = env.reset()
            print('done')
    print(env.observation_space)
    print(env.action_space)
    print(obs.shape)
    print(obs.mean())
    plt.imshow(np.concatenate([obs[:, :, i] for i in range(4)], axis=1), cmap="gray")
    plt.show()
