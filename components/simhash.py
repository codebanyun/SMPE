import numpy as np
import torch
import collections
from gym.spaces.utils import flatdim

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class HashCount:
    """
    Hash-based count bonus for exploration class

    Paper:
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, O. X., Duan, Y., ... & Abbeel, P. (2017).
    # Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in neural information processing systems (pp. 2753-2762).

    Paper: https://arxiv.org/abs/1611.04717

    Open-source code: https://github.com/openai/EPG/blob/master/epg/exploration.py
    """
    def __init__(
        self,
        obs_size,
        observation_space=None,
        action_space=None,
        parallel_envs=None,
        cfg=None,
        **kwargs,
    ):
        """
        Initialise parameters for hash count intrinsic reward
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs: number of parallel environments
        :param config: intrinsic reward configuration dict
        """
        # self.observation_space = observation_space
        # self.action_space = action_space

        self.obs_size = obs_size
        # self.action_size = flatdim(action_space)

        self.parallel_envs = parallel_envs

        # # set all values from config as attributes
        # for k, v in flatten(cfg).items():
        #     setattr(HashCount, k, v)        # Hashing function: SimHash

        self.bucket_sizes = None
        self.key_dim = 16
        self.decay_factor = 1.0
        self.bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        self.bucket_sizes = [9931, 9953, 9959, 9961, 9979, 9983]

        mods_list = []
        for bucket_size in self.bucket_sizes:
            mod = 1
            mods = []
            for _ in range(self.key_dim):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(self.bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))         # hash table
        self.projection_matrix = np.random.normal(size=(self.obs_size, self.key_dim))       # A

    def compute_keys(self, obss):
        if torch.is_tensor(obss):
            obss = obss.detach().cpu().numpy()
        elif isinstance(obss, list) and len(obss) > 0 and torch.is_tensor(obss[0]):
             obss = [o.detach().cpu().numpy() for o in obss]
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast["int"](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, obss):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], self.decay_factor)

    def query_hash(self, obss):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)
        self.inc_hash(obss)

    def predict(self, obs):
        counts = self.query_hash(obs)
        prediction = 1.0 / np.maximum(1.0, np.sqrt(counts))
        return prediction

    def compute_intrinsic_reward(self, state, action=None, next_state=None, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        state = flatten(state)
        state = state.detach().cpu().numpy()
        if train:
            self.fit_before_process_samples(state)
        reward = torch.from_numpy(self.predict(state)).float().to(self.device)

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * reward,
        }

    def reset(self):
        """
        Reset counting
        """
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))



if __name__ == "__main__":

    h = HashCount(obs_size=2)

    # obs = torch.rand((5,4))
    # h.inc_hash(obs)
    # query = h.query_hash(obs)
    # print(h.compute_keys(obs))
    # print(query)

    # obs_ = obs.clone()
    # obs_[0:3,:] += 10
    # h.inc_hash(obs_)  
    # query = h.query_hash(obs)
    # print(query)
    # query = h.query_hash(obs_)
    # print(query)

    # print(obs)
    # print(obs_)


    # print(h.compute_keys(obs))
    # print(h.compute_keys(obs_))


    obs = torch.rand((5,2))
    h.inc_hash(obs)
    rew = h.predict(obs)
    print(rew)
    obs = torch.rand((5,2))
    h.inc_hash(obs)
    rew = h.predict(obs)
    print(rew)
