import numpy as np
import sys
import torch
from utils.common_utils import set_seed
import random
__all__ = ["ReplayBuffer"]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    Implementation of replay buffer with uniform sampling probability.
    """

    def __init__(self, index=0, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 100)
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "obs2": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, self.act_dim), dtype=np.float32
            ),
            "rew": np.zeros(self.max_size, dtype=np.float32),
            "done": np.zeros(self.max_size, dtype=np.float32),
            "logp": np.zeros(self.max_size, dtype=np.float32),
        }
        self.additional_info = kwargs["additional_info"]
        for k, v in self.additional_info.items():
            self.buf[k] = np.zeros(
                combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
            )
            self.buf["next_" + k] = np.zeros(
                combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
            )
        self.ptr, self.size, = (
            0,
            0,
        )

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)

    def store(
        self,
        obs: np.ndarray,
        info: dict,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        logp: np.ndarray,
        next_info: dict,
    ):
        self.buf["obs"][self.ptr] = obs
        self.buf["obs2"][self.ptr] = next_obs
        self.buf["act"][self.ptr] = act
        self.buf["rew"][self.ptr] = rew
        self.buf["done"][self.ptr] = done
        self.buf["logp"][self.ptr] = logp
        for k in self.additional_info.keys():
            self.buf[k][self.ptr] = info[k]
            self.buf["next_" + k][self.ptr] = next_info[k]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples: list):
        for sample in samples:
            self.store(*sample)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {}
        for k, v in self.buf.items():
            batch[k] = v[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
    def add_data_to_buffer(self, dataset):
        n_samples = len(dataset['actions'])  # 或任何一个主键的长度
        for i in range(n_samples):
            obs = dataset['observations'][i]
            next_obs = dataset['next_observations'][i]
            act = dataset['actions'][i]
            rew = dataset['rewards'][i]
            done = dataset['terminals'][i]
            logp = dataset['infos/action_log_probs'][i]
            info = {
                'qpos': dataset['infos/qpos'][i],
                'qvel': dataset['infos/qvel'][i]
            }
            next_info = {
                'qpos': dataset['infos/qpos'][i + 1] if i + 1 < n_samples else dataset['infos/qpos'][i],
                'qvel': dataset['infos/qvel'][i + 1] if i + 1 < n_samples else dataset['infos/qvel'][i]
            }

            self.store(obs, info, act, rew, next_obs, done, logp, next_info)

    def add_random_samples_to_buffer(self, dataset, n):
        n_samples = len(dataset['actions'])  # 确保我们不会抽取超过数据集的样本数量
        n = min(n, n_samples)  # 确保 n 不超过样本总数

        # 随机选择 n 个样本的索引
        indices = random.sample(range(n_samples), n)

        for i in indices:
            obs = dataset['observations'][i]
            next_obs = dataset['next_observations'][i]
            act = dataset['actions'][i]
            rew = dataset['rewards'][i]
            done = dataset['terminals'][i]
            logp = dataset['infos/action_log_probs'][i]
            info = {
                'qpos': dataset['infos/qpos'][i],
                'qvel': dataset['infos/qvel'][i]
            }
            next_info = {
                'qpos': dataset['infos/qpos'][i + 1] if i + 1 < n_samples else dataset['infos/qpos'][i],
                'qvel': dataset['infos/qvel'][i + 1] if i + 1 < n_samples else dataset['infos/qvel'][i]
            }

            self.store(obs, info, act, rew, next_obs, done, logp, next_info)




