__all__ = ["ApproxContainer", "EMDSAC","EMDSACFT"]

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


class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC_V2.

    Contains one policy and a set of Q-value functions.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs["cnn_shared"]:
            feature_args = get_apprfunc_dict("feature", kwargs["value_func_type"], **kwargs)
            kwargs["feature_net"] = create_apprfunc(**feature_args)

        self.num_q_networks = kwargs["num_q_networks"]
        self.q_networks = nn.ModuleList()
        self.q_targets = nn.ModuleList()

        # Create Q networks and targets
        for _ in range(self.num_q_networks):
            q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
            q_network = create_apprfunc(**q_args)
            q_target = deepcopy(q_network)
            self.q_networks.append(q_network)
            self.q_targets.append(q_target)

        # Create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # Set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for q_target in self.q_targets:
            for p in q_target.parameters():
                p.requires_grad = False

        # Create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Create optimizers for Q networks
        self.q_optimizers = []
        for q_network in self.q_networks:
            self.q_optimizers.append(Adam(q_network.parameters(), lr=kwargs["value_learning_rate"]))

        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs["policy_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

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

    def __init__(self, **kwargs):
        super().__init__()
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.num_q_networks = kwargs["num_q_networks"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.mean_stds = [-1.0] * self.num_q_networks
        self.tau_b = kwargs.get("tau_b", self.tau)


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
            f"q{i + 1}_grad": [p._grad for p in q.parameters()]
            for i, q in enumerate(self.networks.q_networks)
        }
        update_info["policy_grad"] = [p._grad for p in self.networks.policy.parameters()]
        update_info["iteration"] = iteration
        if self.auto_alpha:
            update_info["log_alpha_grad"] = self.networks.log_alpha.grad

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        for i, q in enumerate(self.networks.q_networks):
            q_grad = update_info[f"q{i + 1}_grad"]
            for p, grad in zip(q.parameters(), q_grad):
                p._grad = grad

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
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.zero_grad()

        loss_q, q_means, q_stds, q_min_stds = self.__compute_loss_q(data)
        loss_q.backward()

        for q in self.networks.q_networks:
            for p in q.parameters():
                p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for q in self.networks.q_networks:
            for p in q.parameters():
                p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            f"DSACN2/critic_avg_q{i + 1}-RL iter": q_mean.item() for i, q_mean in enumerate(q_means)
        }
        tb_info.update({
            f"DSACN2/critic_avg_std{i + 1}-RL iter": q_std.item() for i, q_std in enumerate(q_stds)
        })
        tb_info.update({
            f"DSACN2/critic_avg_min_std{i + 1}-RL iter": q_min_std.item() for i, q_min_std in enumerate(q_min_stds)
        })
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

        q_means, q_stds = [], []
        q_next_means, q_next_samples = [], []

        for q in self.networks.q_networks:
            q_mean, q_std, _ = self.__q_evaluate(obs, act, q)
            q_means.append(q_mean)
            q_stds.append(q_std)

        for q_target in self.networks.q_targets:
            q_next_mean, _, q_next_sample = self.__q_evaluate(obs2, act2, q_target)
            q_next_means.append(q_next_mean)
            q_next_samples.append(q_next_sample)

        for i in range(self.num_q_networks):
            if self.mean_stds[i] == -1.0:
                self.mean_stds[i] = torch.mean(q_stds[i].detach())
            else:
                self.mean_stds[i] = (1 - self.tau_b) * self.mean_stds[i] + self.tau_b * torch.mean(q_stds[i].detach())

        q_next_min = torch.min(torch.stack(q_next_means), dim=0).values
        q_next_sample_min = torch.where(q_next_means[0] < q_next_means[1], q_next_samples[0], q_next_samples[1])
        for i in range(2, len(q_next_means)):
            q_next_min = torch.min(q_next_means[i], q_next_min)
            q_next_sample_min = torch.where(q_next_means[i] < q_next_sample_min, q_next_samples[i], q_next_sample_min)
        q_next = q_next_min
        q_next_sample = q_next_sample_min

        target_qs, target_q_bounds = [], []
        for i, q_mean in enumerate(q_means):
            target_q, target_q_bound = self.__compute_target_q(
                rew,
                done,
                q_mean.detach(),
                self.mean_stds[i].detach(),
                q_next.detach(),
                q_next_sample.detach(),
                log_prob_act2.detach(),
            )
            target_qs.append(target_q)
            target_q_bounds.append(target_q_bound)

        q_losses = []
        for i, q_std in enumerate(q_stds):
            q_std_detach = torch.clamp(q_std, min=0.).detach()
            bias = 0.1
            q_loss = (torch.pow(self.mean_stds[i], 2) + bias) * torch.mean(
                -(target_qs[i] - q_means[i]).detach() / (torch.pow(q_std_detach, 2) + bias) * q_means[i]
                - ((torch.pow(q_means[i].detach() - target_q_bounds[i], 2) - q_std_detach.pow(2)) / (
                        torch.pow(q_std_detach, 3) + bias)
                   ) * q_std
            )
            q_losses.append(q_loss)

        return sum(q_losses), \
            [q_mean.detach().mean() for q_mean in q_means], \
            [q_std.detach().mean() for q_std in q_stds], \
            [q_std.min().detach() for q_std in q_stds]

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
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q_values = [self.__q_evaluate(obs, new_act, q)[0] for q in self.networks.q_networks]
        q_min = torch.min(torch.stack(q_values), dim=0).values
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

    def __update(self, iteration: int):
        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for q, q_target in zip(self.networks.q_networks, self.networks.q_targets):
                    for p, p_targ in zip(q.parameters(), q_target.parameters()):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(
                        self.networks.policy.parameters(),
                        self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)


class EMDSACFT:
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
        self.lamda=kwargs["lamda"]
        self.ep_min=kwargs["ep_min"]
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.mean_stds = [-1.0] * self.num_q_networks
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.it=0
        self.ra=1.0
        self.det=kwargs["det"]

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
            f"q{i + 1}_grad": [p._grad for p in q.parameters()]
            for i, q in enumerate(self.networks.q_networks)
        }
        update_info["policy_grad"] = [p._grad for p in self.networks.policy.parameters()]
        update_info["iteration"] = iteration
        if self.auto_alpha:
            update_info["log_alpha_grad"] = self.networks.log_alpha.grad

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        for i, q in enumerate(self.networks.q_networks):
            q_grad = update_info[f"q{i + 1}_grad"]
            for p, grad in zip(q.parameters(), q_grad):
                p._grad = grad

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
        data.update({"new_act": new_act, "new_log_prob": new_log_prob,"logits_mean":logits_mean,"logits_std":logits_std})

        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.zero_grad()

        loss_q, q_means, q_stds, q_min_stds,bellman_q_loss_mean,bellman_q_loss_std,bellman_q_loss_max,bellman_q_loss_min = self.__compute_loss_q(data)
        loss_q.backward()

        for q in self.networks.q_networks:
            for p in q.parameters():
                p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for q in self.networks.q_networks:
            for p in q.parameters():
                p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            f"DSACN2/critic_avg_q{i + 1}-RL iter": q_mean.item() for i, q_mean in enumerate(q_means)
        }
        tb_info.update({
            f"DSACN2/critic_avg_std{i + 1}-RL iter": q_std.item() for i, q_std in enumerate(q_stds)
        })
        tb_info.update({
            f"DSACN2/critic_avg_min_std{i + 1}-RL iter": q_min_std.item() for i, q_min_std in enumerate(q_min_stds)
        })
        tb_info.update({
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "DSACN2/policy_mean-RL iter": policy_mean,
            "DSACN2/policy_std-RL iter": policy_std,
            "DSACN2/entropy-RL iter": entropy.item(),
            "DSACN2/alpha-RL iter": self.__get_alpha(),
            "DSACN2/bellman_q_Loss_mean iter": bellman_q_loss_mean,
            "DSACN2/bellman_q_Loss_std": bellman_q_loss_std,
            "DSACN2/bellman_q_Loss_max": bellman_q_loss_max,
            "DSACN2/bellman_q_Loss_min": bellman_q_loss_min,


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
        noise = (torch.randn_like(act2) *0.15).clamp(-0.2, 0.2)
        #noise = (torch.randn_like(act2) * 0.2).clamp(-0.3, 0.3)#key与10搭配
        #noise = (torch.randn_like(act2) * 0.05).clamp(-0.1, 0.1)  # key与10搭配
        act2 = (act2+ noise).clamp(-1.0, 1.0)


        q_mean1, q_std1, _ = self.__q_evaluate(obs, act, self.networks.q_networks[0])
        q_mean2, q_std2, _ = self.__q_evaluate(obs, act, self.networks.q_networks[1])


        q_next_mean1, _, q_next_sample1 = self.__q_evaluate(obs2, act2, self.networks.q_targets[0])
        q_next_mean2, _, q_next_sample2 = self.__q_evaluate(obs2, act2, self.networks.q_targets[1])



        if self.mean_stds[0] == -1.0:
            self.mean_stds[0] = torch.mean(q_std1.detach())
        else:
            self.mean_stds[0] = (1 - self.tau_b) * self.mean_stds[0] + self.tau_b * torch.mean(q_std1.detach())

        if self.mean_stds[1] == -1.0:
            self.mean_stds[1] = torch.mean(q_std2.detach())
        else:
            self.mean_stds[1] = (1 - self.tau_b) * self.mean_stds[1] + self.tau_b * torch.mean(q_std2.detach())


        q_next_min = torch.min(q_next_mean1, q_next_mean2)
        q_next_sample_min = torch.where(q_next_mean1 < q_next_mean2, q_next_sample1, q_next_sample2)

        q_next = q_next_min
        q_next_sample = q_next_sample_min


        target_q1, target_q_bound1 = self.__compute_target_q(
            rew,
            done,
            q_mean1.detach(),
            self.mean_stds[0].detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        target_q2, target_q_bound2 = self.__compute_target_q(
            rew,
            done,
            q_mean2.detach(),
            self.mean_stds[1].detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q_std1_detach = torch.clamp(q_std1, min=0.).detach()
        bias = 0.1


        q1_loss = (torch.pow(self.mean_stds[0], 2) + bias) * torch.mean((
            -(target_q1 - q_mean1).detach() / (torch.pow(q_std1_detach, 2) + bias) * q_mean1
            - ((torch.pow(q_mean1.detach() - target_q_bound1, 2) - q_std1_detach.pow(2)) / (
                    torch.pow(q_std1_detach, 3) + bias)
               ) * q_std1
        ))


        q_std2_detach = torch.clamp(q_std2, min=0.).detach()
        bias = 0.1
        q2_loss = (torch.pow(self.mean_stds[1], 2) + bias) * torch.mean((
            -(target_q2 - q_mean2).detach() / (torch.pow(q_std2_detach, 2) + bias) * q_mean2
            - ((torch.pow(q_mean2.detach() - target_q_bound2, 2) - q_std2_detach.pow(2)) / (
                    torch.pow(q_std2_detach, 3) + bias)
               ) * q_std2
        ))
        bellman_q_loss=(torch.pow((target_q2 - q_mean2),2)+torch.pow((target_q1 - q_mean1),2))/2#(bs,)
        data.update({"bellman_q_loss": bellman_q_loss})
        return q1_loss+q2_loss, \
            [q_mean1.detach().mean(),q_mean2.detach().mean()], \
            [q_std1.detach().mean(),q_std2.detach().mean()], \
            [q_std1.min().detach(),q_std2.min().detach()], \
            bellman_q_loss.mean().detach(),bellman_q_loss.std().detach(),bellman_q_loss.max().detach(),bellman_q_loss.min().detach()

    def com_loss_q(self, obs, act, rew, obs2, done):

        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q_mean1, q_std1, _ = self.__q_evaluate(obs, act, self.networks.q_networks[0])
        q_mean2, q_std2, _ = self.__q_evaluate(obs, act, self.networks.q_networks[1])

        q_next_mean1, _, q_next_sample1 = self.__q_evaluate(obs2, act2, self.networks.q_targets[0])
        q_next_mean2, _, q_next_sample2 = self.__q_evaluate(obs2, act2, self.networks.q_targets[1])


        if self.mean_stds[0] == -1.0:
            self.mean_stds[0] = torch.mean(q_std1.detach())
        else:
            self.mean_stds[0] = (1 - self.tau_b) * self.mean_stds[0] + self.tau_b * torch.mean(q_std1.detach())

        if self.mean_stds[1] == -1.0:
            self.mean_stds[1] = torch.mean(q_std2.detach())
        else:
            self.mean_stds[1] = (1 - self.tau_b) * self.mean_stds[1] + self.tau_b * torch.mean(q_std2.detach())


        q_next_min = torch.min(q_next_mean1, q_next_mean2)
        q_next_sample_min = torch.where(q_next_mean1 < q_next_mean2, q_next_sample1, q_next_sample2)

        q_next = q_next_min
        q_next_sample = q_next_sample_min

        target_q1, target_q_bound1 = self.__compute_target_q(
            rew,
            done,
            q_mean1.detach(),
            self.mean_stds[0].detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        target_q2, target_q_bound2 = self.__compute_target_q(
            rew,
            done,
            q_mean2.detach(),
            self.mean_stds[1].detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q_std1_detach = torch.clamp(q_std1, min=0.).detach()
        bias = 0.1

        ratio1=torch.pow((target_q1 - q_mean1),2)+1
        ratio2 = torch.pow((target_q2 - q_mean2), 2)+1

        q1_loss = (torch.pow(self.mean_stds[0], 2) + bias) * torch.mean((
            -(target_q1 - q_mean1).detach() / (torch.pow(q_std1_detach, 2) + bias) * q_mean1
            - ((torch.pow(q_mean1.detach() - target_q_bound1, 2) - q_std1_detach.pow(2)) / (
                    torch.pow(q_std1_detach, 3) + bias)
               ) * q_std1
        ))


        q_std2_detach = torch.clamp(q_std2, min=0.).detach()
        bias = 0.1
        q2_loss = (torch.pow(self.mean_stds[1], 2) + bias) * torch.mean((
            -(target_q2 - q_mean2).detach() / (torch.pow(q_std2_detach, 2) + bias) * q_mean2
            - ((torch.pow(q_mean2.detach() - target_q_bound2, 2) - q_std2_detach.pow(2)) / (
                    torch.pow(q_std2_detach, 3) + bias)
               ) * q_std2
        ))
        bellman_q_loss=(torch.pow((target_q2 - q_mean2),2)+torch.pow((target_q1 - q_mean1),2))/2#(bs,)

        return q1_loss+q2_loss, \
            [q_mean1.detach().mean(),q_mean2.detach().mean()], \
            [q_std1.detach().mean(),q_std2.detach().mean()], \
            [q_std1.min().detach(),q_std2.min().detach()], \
            bellman_q_loss.mean().detach(),bellman_q_loss.std().detach(),bellman_q_loss.max().detach(),bellman_q_loss.min().detach()

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
        self.it=self.it+1
        obs, new_act, new_log_prob, logits_mean, logits_std, act,bellman_q_loss = data["obs"], data["new_act"], data["new_log_prob"], \
        data["logits_mean"], data["logits_std"], data["act"],data["bellman_q_loss"]

        q1, _, _ = self.__q_evaluate(obs, new_act, self.networks.q_networks[0])
        q2, _, _ = self.__q_evaluate(obs, new_act, self.networks.q_networks[1])


        loss_policy = (-self.__get_alpha() * new_log_prob + torch.min(q1, q2))
        entropy = -new_log_prob.detach().mean()
        lmbda = 10 / loss_policy.abs().mean().detach()

        if self.it%5000==0:
            print("q_mean",torch.mean(torch.min(q1, q2)))
            print("bellman_q_loss.mean()", bellman_q_loss.mean())
                    
        if self.it%10000==0:
            self.ra=self.ra*1
            
            
        
        

        logits_ = self.networks.policy_target(obs)
        act_dist = self.networks.create_action_distributions(logits_)
        act_, _ = act_dist.rsample()
        
        if self.det:
            bc_loss=torch.mean(torch.pow((act-torch.tanh(logits_mean)),2))
        else:
            bc_loss=torch.mean((torch.pow((act-torch.tanh(logits_mean)),2))/torch.pow(logits_std,2)+(torch.log(logits_std)),dim=1)

        
        ratio=torch.clamp(self.lamda*bellman_q_loss.mean().detach()-self.lamda*self.ep_min,0,1.25)
        
        loss=self.ra*ratio*bc_loss.mean()-lmbda*loss_policy.mean()

            


        return loss, entropy

    def __compute_loss_alpha(self, data: Dict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
                -self.networks.log_alpha
                * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.step()
        
        if self.it % 2 == 0:
            with torch.no_grad():
                polyak = 1 - self.tau
                for q, q_target in zip(self.networks.q_networks, self.networks.q_targets):
                    for p, p_targ in zip(q.parameters(), q_target.parameters()):
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
        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.step()

        if self.it % 2 == 0:
            with torch.no_grad():
                polyak = 1 - self.tau
                for q, q_target in zip(self.networks.q_networks, self.networks.q_targets):
                    for p, p_targ in zip(q.parameters(), q_target.parameters()):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)


    def local_update_wo_policy(self, data: Dict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update_wo_policy(iteration)
        return tb_info


    def value_update(self, data: Dict):

        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.zero_grad()

        loss_q, q_means, q_stds, q_min_stds,bellman_q_loss_mean,bellman_q_loss_std,bellman_q_loss_max,bellman_q_loss_min = self.__compute_loss_q(data)
        loss_q.backward()

        for q_optimizer in self.networks.q_optimizers:
            q_optimizer.step()
