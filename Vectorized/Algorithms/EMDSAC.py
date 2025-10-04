
import time
from copy import deepcopy
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam


import copy
import math
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.independent import Independent
from torch.distributions.transforms import AffineTransform

'''
Original SAC-N code - https://github.com/snu-mllab/EDAC
Vectorised Linear code - https://github.com/tinkoff-ai/CORL/blob/main/algorithms/sac_n.py
'''


class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_critics=2, hidden_dim=256):
        super(VectorizedCritic, self).__init__()

        self.l1 = VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics)
        self.l2 = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.l3 = VectorizedLinear(hidden_dim, hidden_dim, num_critics)
        self.qs = VectorizedLinear(hidden_dim, 2, num_critics)

        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = F.relu(self.l1(state_action))
        q_values = F.relu(self.l2(q_values))
        q_values = F.relu(self.l3(q_values))
        q_values = self.qs(q_values).squeeze(-1)
        value_mean, value_std = q_values[..., 0], q_values[..., -1]

        value_log_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean.unsqueeze(dim=2), value_log_std.unsqueeze(dim=2)), dim=-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, hidden_dim=256, log_std_min=-20.0,
                 log_std_max=2.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.log_std_min = log_std_min  # for numerical stability
        self.log_std_max = log_std_max  # for numerical stability
        self.action_mean = (max_action + min_action) / 2  # to allow for action domains other than [-1, 1]
        self.action_scale = (max_action - min_action) / 2  # to allow for action domains other than [-1, 1]
        self.min_action = min_action
        self.max_action = max_action
        self.gelu = nn.GELU()
        self.eps = 1e-6

    def forward(self, state):
        a = self.gelu(self.l1(state))
        a = self.gelu(self.l2(a))
        a = self.gelu(self.l3(a))
        mean = self.mean(a)
        std = self.log_std(a).clamp(self.log_std_min, self.log_std_max).exp()

        return mean, std

    def sample_normal(self, state):
        mu, sigma = self.forward(state)
        dist = TransformedDistribution(Independent(Normal(mu, sigma), 1), [
            TanhTransform(cache_size=1), AffineTransform(self.action_mean, self.action_scale, cache_size=1)])
        actions = dist.rsample()  # For repam trick
        log_prob_action = dist.log_prob(actions)

        return actions, log_prob_action


class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC_V2.

    Contains one policy and a set of Q-value functions.
    """

    def __init__(self, obsv_dim, action_dim, min_action, max_action,num_q_networks,device,**kwargs):
        super().__init__()

        self.q_networks = VectorizedCritic(obsv_dim, action_dim, num_critics=num_q_networks).to(
            f'cuda:{device}')
        self.q_targets = deepcopy(self.q_networks)
        # Create policy network
        self.policy=Actor(obsv_dim, action_dim, min_action, max_action).to(
            f'cuda:{device}')
        self.policy_target = deepcopy(self.policy)

        # Set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False

        for p in self.q_targets.parameters():
            p.requires_grad = False

        # Create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.q_optimizers = Adam(self.q_networks.parameters(), lr=3e-4)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=3e-4)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class EMDSAC:
    """DSAC_V2(DSAC-T) algorithm

    Paper: https://arxiv.org/abs/2310.05858

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param float delay_update: delay update steps for actor.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(self, obsv_dim, action_dim, min_action, max_action,num_q_networks,device,save_plot,gamma=0.99,tau=0.005,):
        super().__init__()
        self.networks = ApproxContainer(obsv_dim, action_dim, min_action, max_action,num_q_networks,device)
        self.num_q_networks = num_q_networks
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim
        self.auto_alpha = True
        self.alpha = 0.2
        self.delay_update = 2
        self.mean_stds = [-1.0] * self.num_q_networks
        self.device=f'cuda:{device}'
        self.mean_stdss = torch.full((num_q_networks,), -1.0, device=self.device)
        self.tau_b = tau
        self.it = 0
        self.max_action=max_action
        self.save_plot=save_plot


    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def local_update(self, data: Dict) -> dict:
        tb_info = self.__compute_gradient(data)
        self.__update()
        return tb_info

    def choose_action(self, state, mean=True):
        # Take mean/greedy action by default, but also allows sampling from policy
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if mean == True:
                action = self.max_action * torch.tanh(self.networks.policy(state)[0])
            else:
                action = self.networks.policy.sample_normal(state)[0]

        return action.cpu().numpy().flatten()

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            return alpha if requires_grad else alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: Dict):
        self.it += 1
        start_time = time.time()

        obs = data["obs"]

        logits_mean, logits_std = self.networks.policy(obs)
        #print(logits_mean, logits_std)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        new_act, new_log_prob = self.networks.policy.sample_normal(obs)

        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        self.networks.q_optimizers.zero_grad()

        if self.save_plot:
            loss_q,bellman_loss, q_mean_mean, q_mean_std, q_std_mean, q_std_std,ui,uo,ur,ri,ro,rr = self.__compute_loss_q(data)
        else:
            loss_q,q_mean_mean, q_mean_std,q_std_mean, q_std_std = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q_networks.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q_networks.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            f"EMDSAC/critic_mean_q_mean_-RL iter": q_mean_mean.item(),
            f"EMDSAC/critic_mean_q_std_-RL iter": q_mean_std.item(),
            f"EMDSAC/critic_std_q_mean_-RL iter": q_std_mean.item(),
            f"EMDSAC/critic_std_q_std_-RL iter": q_std_std.item(),
            f"EMDSAC/ui-RL iter": ui.item(),
            f"EMDSAC/uo-RL iter": uo.item(),
            f"EMDSAC/ur-RL iter": ur.item(),
            f"EMDSAC/ri-RL iter": ri.item(),
            f"EMDSAC/ro-RL iter": ro.item(),
            f"EMDSAC/rr-RL iter": rr.item(),
            f"EMDSAC/bellman_loss": bellman_loss.item(),
        }

        tb_info.update({
            "loss_actor": loss_policy.item(),
            "loss_critic": loss_q.item(),
            "EMDSAC/policy_mean-RL iter": policy_mean,
            "EMDSAC/policy_std-RL iter": policy_std,
            "EMDSAC/entropy-RL iter": entropy.item(),
            "EMDSAC/alpha-RL iter": self.__get_alpha(),
            "alg_time": (time.time() - start_time) * 1000,
        })


        return tb_info

    def __q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data: Dict):
        obs, act, rew, obs2, done = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        act2, log_prob_act2 = self.networks.policy.sample_normal(obs2)

        q_means, q_stds, _ = self.__q_evaluate(obs, act, self.networks.q_networks)

        if self.it % 5000 == 0:
            print("q_means", q_means.mean())

        q_next_means, _, q_next_samples = self.__q_evaluate(obs2, act2, self.networks.q_targets)


        if torch.all(self.mean_stdss == torch.full((self.num_q_networks,), -1.0, device=self.device)):
            self.mean_stdss = torch.mean(q_stds.detach(), dim=-1)
        else:
            self.mean_stdss = (1 - self.tau) * self.mean_stdss + self.tau * torch.mean(q_stds.detach(), dim=1)



        q_next_min = torch.min(q_next_means, dim=0)


        q_next = q_next_min.values
        q_next_sample = q_next_samples[q_next_min.indices, torch.arange(q_means.shape[1])]

        target_qs, target_q_bounds = self.__compute_target_q__(
            rew,
            done,
            q_means.detach(),
            self.mean_stdss.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        target_qs=target_qs.expand(target_q_bounds.shape[0], -1)

        q_std_detach = torch.clamp(q_stds, min=0.).detach()
        bias = 0.1
        bellman_loss = ((q_means.detach() - target_qs.detach()) ** 2).mean(dim=1).sum(dim=0)
        q_loss = torch.sum((torch.pow(self.mean_stdss, 2) + bias) * torch.mean(
                -(target_qs - q_means).detach() / (torch.pow(q_std_detach, 2) + bias) * q_means
                - ((torch.pow(q_means.detach() - target_q_bounds, 2) - q_std_detach.pow(2)) / (
                        torch.pow(q_std_detach, 3) + bias)
                   ) * q_stds, dim=-1))
                   

        if self.save_plot:
            uncertainty_in, uncertainty_ood, uncertainty_random, randomness_in, randomness_ood, randomness_random=self.cal_ur(obs, act,obs2,act2)
            return (q_loss,bellman_loss.detach().mean(), q_means.detach().mean(), q_means.detach().std(), q_stds.detach().mean(), q_stds.detach().std()
                        ,uncertainty_in.detach().mean(), uncertainty_ood.detach().mean(), uncertainty_random.detach().mean()
                        , randomness_in.detach().mean(), randomness_ood.detach().mean(), randomness_random.detach().mean())
        else:
            return q_loss, q_means.detach().mean(), q_means.detach().std(), q_stds.detach().mean(), q_stds.detach().std()


    def cal_ur(self, obs, act,obs2,act2):

        random_act=2*torch.rand(act.shape,device=act.device)-1



        q_means, q_stds, _ = self.__q_evaluate(obs, act, self.networks.q_networks)
        q_ood_means, q_ood_stds, _ = self.__q_evaluate(obs2, act2, self.networks.q_networks)
        q_random_means, q_random_stds, _ = self.__q_evaluate(obs2, random_act, self.networks.q_networks)




        uncertainty_in=torch.mean(q_means, dim=0)-torch.min(q_means,dim=0).values
        uncertainty_ood = torch.mean(q_ood_means, dim=0) - torch.min(q_ood_means, dim=0).values
        uncertainty_random = torch.mean(q_random_means, dim=0) - torch.min(q_random_means, dim=0).values

        randomness_in=torch.mean(q_stds, dim=0)
        randomness_ood = torch.mean(q_ood_stds, dim=0)
        randomness_random = torch.mean(q_random_stds, dim=0)

        return uncertainty_in,uncertainty_ood,uncertainty_random,randomness_in,randomness_ood,randomness_random


    def __compute_target_q(self, r, done, q, q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
                q_next - self.__get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
                q_next_sample - self.__get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __compute_target_q__(self, r, done, q, q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
                q_next - self.__get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
                q_next_sample - self.__get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        abs_td_bound = torch.abs(td_bound)
        abs_td_bound_expanded = abs_td_bound.view(-1, 1)

        difference = (target_q_sample - q).clamp(min=-abs_td_bound_expanded, max=abs_td_bound_expanded)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()


    def __compute_loss_policy(self, data: Dict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q_values = self.__q_evaluate(obs, new_act, self.networks.q_networks)[0]
        q_min = torch.min(q_values, dim=0).values
        loss_policy = (self.__get_alpha() * new_log_prob - q_min).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data: Dict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
                -self.networks.log_alpha
                * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self):
        self.networks.q_optimizers.step()

        if self.it % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(self.networks.q_networks.parameters(), self.networks.q_targets.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(
                        self.networks.policy.parameters(),
                        self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def save_checkpoint(self, path: str):
        """保存模型参数和优化器状态"""
        checkpoint = {
            # 策略网络参数
            'policy_state_dict': self.networks.policy.state_dict(),
            'policy_target_state_dict': self.networks.policy_target.state_dict(),

            # Q网络参数
            'q_networks_state_dict': self.networks.q_networks.state_dict(),
            'q_targets_state_dict': self.networks.q_targets.state_dict(),

            # 优化器状态
            'policy_optimizer': self.networks.policy_optimizer.state_dict(),
            'q_optimizers': self.networks.q_optimizers.state_dict(),
            'alpha_optimizer': self.networks.alpha_optimizer.state_dict(),

            # 自动温度参数
            'log_alpha': self.networks.log_alpha,

            # 训练状态
            'it': self.it,
            'mean_stdss': self.mean_stdss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: str = None):
        """加载模型参数和优化器状态"""
        checkpoint = torch.load(path, map_location=device)

        # 加载网络参数
        self.networks.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.networks.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.networks.q_networks.load_state_dict(checkpoint['q_networks_state_dict'])
        self.networks.q_targets.load_state_dict(checkpoint['q_targets_state_dict'])

        # 加载优化器状态
        self.networks.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.networks.q_optimizers.load_state_dict(checkpoint['q_optimizers'])
        self.networks.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        # 加载训练状态
        self.networks.log_alpha = checkpoint['log_alpha']
        self.it = checkpoint['it']
        self.mean_stdss = checkpoint['mean_stdss']

        # 确保加载后目标网络不需要梯度
        for p in self.networks.policy_target.parameters():
            p.requires_grad = False
        for p in self.networks.q_targets.parameters():
            p.requires_grad = False

    def load_partial_checkpoint(self, path: str, device: str = None):

        checkpoint = torch.load(path, map_location=device)

        # 加载策略网络参数（保持不变）
        self.networks.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.networks.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])

        # 自定义Q网络参数加载
        def load_partial_q(src_state_dict, dest_network):
            new_state_dict = {}
            for key in src_state_dict:
                # 只处理集成线性层的weight和bias参数
                if "l1.weight" in key or "l1.bias" in key or \
                        "l2.weight" in key or "l2.bias" in key or \
                        "l3.weight" in key or "l3.bias" in key or \
                        "qs.weight" in key or "qs.bias" in key:

                    param = src_state_dict[key]
                    # 取前两个集成成员的参数
                    sliced_param = param[:2]  # 关键切片操作
                    new_state_dict[key] = sliced_param
                else:
                    new_state_dict[key] = src_state_dict[key]
            dest_network.load_state_dict(new_state_dict, strict=False)

        # 加载当前Q网络参数
        load_partial_q(checkpoint['q_networks_state_dict'], self.networks.q_networks)

        # 加载目标Q网络参数
        load_partial_q(checkpoint['q_targets_state_dict'], self.networks.q_targets)

        # 加载其他参数
        self.networks.log_alpha = checkpoint['log_alpha']
        self.it = checkpoint['it']

        # 注意：这里不加载原优化器状态，因为网络结构已变化
        # 重新初始化优化器
        self.networks.q_optimizers = Adam(self.networks.q_networks.parameters(), lr=3e-4)
        # 加载优化器状态
        self.networks.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.networks.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        self.mean_stdss = checkpoint['mean_stdss'][:2]


        print(self.mean_stdss)
        # 确保目标网络不需要梯度
        for p in self.networks.policy_target.parameters():
            p.requires_grad = False
        for p in self.networks.q_targets.parameters():
            p.requires_grad = False

