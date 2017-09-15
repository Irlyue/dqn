import numpy as np


from collections import deque


class ReplayBuffer(object):
    def __init__(self, maxlen):
        """
        :param maxlen: size of the buffer
        """
        self.maxlen = maxlen
        self._storage = deque(maxlen=maxlen)

    def __str__(self):
        return "ReplayBuffer(maxlen=%d,)" % (self.maxlen,)

    def add(self, obs_t1, action, reward, obs_tp1, done):
        data = (obs_t1, action, reward, obs_tp1, done)
        self._storage.append(data)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self._storage), size=batch_size)
        return self._sample_given_indexes(indexes)

    def _sample_given_indexes(self, indexes):
        obs_t1s, actions, rewards, obs_tpls, dones = [], [], [], [], []
        for idx in indexes:
            data = self._storage[idx]
            obs_t1, action, reward, obs_tpl, done = data
            obs_t1s.append(np.array(obs_t1, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            obs_tpls.append(np.array(obs_tpl, copy=False))
            dones.append(np.array(done, copy=False))
        return [np.array(item) for item in (obs_t1s, actions, rewards, obs_tpls, dones)]


def test():
    """Simple test"""
    buffer = ReplayBuffer(maxlen=5)
    for i in range(6):
        data = [i + di for di in range(5)]
        buffer.add(*data)
    print(buffer)
    print(buffer._sample_given_indexes(range(5)))


if __name__ == '__main__':
    test()