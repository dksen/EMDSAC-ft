__all__ = ["ApproxContainer", "EMDSACF"]

import time
from copy import deepcopy
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from utils.tensorboard_setup import tb_tags
from utils.initialization import create_apprfunc
from utils.common_utils import get_apprfunc_dict

import math
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
        self.gelu = nn.GELU()

        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        q_values = self.gelu(self.l1(state_action))
        q_values = self.gelu(self.l2(q_values))
        q_values = self.gelu(self.l3(q_values))
        q_values = self.qs(q_values).squeeze(-1)
        value_mean, value_std = q_values[..., 0], q_values[..., -1]

        value_log_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean.unsqueeze(dim=2), value_log_std.unsqueeze(dim=2)), dim=-1)

class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC_V2.

    Contains one policy and a set of Q-value functions.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs["cnn_shared"]:
            feature_args = get_apprfunc_dict("feature", kwargs["value_func_type"], **kwargs)
            kwargs["feature_net"] = create_apprfunc(**feature_args)

        # self.num_q_networks = kwargs["num_q_networks"]
        # self.q_networks = nn.ModuleList()
        # self.q_targets = nn.ModuleList()

        self.q_networks = VectorizedCritic(kwargs["obsv_dim"], kwargs["action_dim"],
                                           num_critics=kwargs["num_q_networks"]).to(
            f'cuda:{kwargs["device"]}')
        self.q_targets = deepcopy(self.q_networks)

        # # Create Q networks and targets
        # for _ in range(self.num_q_networks):
        #     q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        #     q_network = create_apprfunc(**q_args)
        #     q_target = deepcopy(q_network)
        #     self.q_networks.append(q_network)
        #     self.q_targets.append(q_target)

        # Create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # Set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for q in self.q_targets.parameters():
            q.requires_grad = False
        # for q_target in self.q_targets:
        #     for p in q_target.parameters():
        #         p.requires_grad = False

        # Create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Create optimizers for Q networks
        #self.q_optimizers = []
        # for q_network in self.q_networks:
        #     self.q_optimizers.append(Adam(q_network.parameters(), lr=kwargs["value_learning_rate"]))
        self.q_optimizers = (Adam(self.q_networks.parameters(), lr=kwargs["value_learning_rate"]))

        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class EMDSACF:
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

    def __init__(self, **kwargs):
        super().__init__()
        self.networks = ApproxContainer(**kwargs)
        self.num_q_networks = kwargs["num_q_networks"]
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.mean_stds = [-1.0] * self.num_q_networks
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.it = 0

        self.device = f'cuda:{kwargs["device"]}'

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def local_update(self, data: Dict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(self, data: Dict, iteration: int) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            f"q_grad": [p._grad for p in self.networks.q_networks.parameters()]

        }
        update_info["policy_grad"] = [p._grad for p in self.networks.policy.parameters()]
        update_info["iteration"] = iteration
        if self.auto_alpha:
            update_info["log_alpha_grad"] = self.networks.log_alpha.grad

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]

        q_grad = update_info["q_grad"]
        for q, grad in zip(self.networks.policy.parameters(), q_grad):
            q._grad = grad

        policy_grad = update_info["policy_grad"]
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            return alpha if requires_grad else alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: Dict, iteration: int):
        start_time = time.time()

        obs = data["obs"]

        logits = self.networks.policy(obs)
        # print(logits)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update(
            {"new_act": new_act, "new_log_prob": new_log_prob, "logits_mean": logits_mean, "logits_std": logits_std})

        self.networks.q_optimizers.zero_grad()

        loss_q, q_mean_mean, q_mean_std, q_std_mean, q_std_std, bellman_q_loss = self.__compute_loss_q(
            data)
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
            f"DSACN2/critic_mean_q_mean_-RL iter": q_mean_mean.item(),
            f"DSACN2/critic_mean_q_std_-RL iter": q_mean_std.item(),
            f"DSACN2/critic_std_q_mean_-RL iter": q_std_mean.item(),
            f"DSACN2/critic_std_q_std_-RL iter": q_std_std.item(),
        }
        tb_info.update({
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "DSACN2/policy_mean-RL iter": policy_mean,
            "DSACN2/policy_std-RL iter": policy_std,
            "DSACN2/entropy-RL iter": entropy.item(),
            "DSACN2/alpha-RL iter": self.__get_alpha(),

            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
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
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()
        noise = (torch.randn_like(act2) * 0.05).clamp(-0.1, 0.1)
        act2 = act + noise



        q_means, q_stds, _ = self.__q_evaluate(obs, act, self.networks.q_networks)
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

        target_qs = target_qs.expand(target_q_bounds.shape[0], -1)

        q_std_detach = torch.clamp(q_stds, min=0.).detach()
        bias = 0.1
        q_loss = torch.sum((torch.pow(self.mean_stdss, 2) + bias) * torch.mean(
            -(target_qs - q_means).detach() / (torch.pow(q_std_detach, 2) + bias) * q_means
            - ((torch.pow(q_means.detach() - target_q_bounds, 2) - q_std_detach.pow(2)) / (
                    torch.pow(q_std_detach, 3) + bias)
               ) * q_stds, dim=-1))

        bellman_q_loss = torch.pow((target_qs - q_means), 2)
        data.update({"bellman_q_loss": bellman_q_loss})
        return q_loss, q_means.detach().mean(), q_means.detach().std(), q_stds.detach().mean(), q_stds.detach().std(), bellman_q_loss.mean().detach()

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

    def __compute_loss_policy(self, data: Dict):
        self.it = self.it + 1
        obs, new_act, new_log_prob, logits_mean, logits_std, act, bellman_q_loss = data["obs"], data["new_act"], data[
            "new_log_prob"], \
            data["logits_mean"], data["logits_std"], data["act"], data["bellman_q_loss"]


        q_values = self.__q_evaluate(obs, new_act, self.networks.q_networks)[0]
        q_min = torch.min(q_values, dim=0).values
        loss_policy = (-self.__get_alpha() * new_log_prob + q_min)

        entropy = -new_log_prob.detach().mean()
        lmbda = 1000 / loss_policy.abs().mean().detach()


        loss = - lmbda * loss_policy.mean()

        return loss, entropy

    def __compute_loss_alpha(self, data: Dict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
                -self.networks.log_alpha
                * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.q_optimizers.step()

        if self.it % 2 == 0:
            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(self.networks.q_networks.parameters(), self.networks.q_targets.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        if self.it % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                for p, p_targ in zip(
                        self.networks.policy.parameters(),
                        self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def __update_wo_policy(self, iteration: int):
        self.networks.q_optimizers.step()

        if self.it % 2 == 0:
            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(self.networks.q_networks.parameters(), self.networks.q_targets.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def local_update_wo_policy(self, data: Dict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update_wo_policy(iteration)
        return tb_info

    def value_update(self, data: Dict):

        self.networks.q_optimizers.zero_grad()

        self.networks.q_optimizers.zero_grad()
        loss_q, q_mean_mean, q_mean_std, q_std_mean, q_std_std, bellman_q_loss_mean = self.__compute_loss_q(data)
        loss_q.backward()

        self.networks.q_optimizers.step()

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
        """加载部分Q网络参数（前两个集成成员）"""
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
