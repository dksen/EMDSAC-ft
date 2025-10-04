import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import csv
from datetime import datetime

import gym
import random
import numpy as np
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from env_replay_buffer import EnvReplayBuffer

from Algorithms import EMDSAC

import d4rl


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed=0, seed_offset=100, eval_episodes=10, mean=0, std=1):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.array(state).reshape(1, -1)
            state = (state - mean) / std
            action = policy.choose_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return avg_reward, d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="EMDSAC")  # Online policy name
    parser.add_argument("--env", default="hopper-expert-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default=False)  # Whether load offline model
    parser.add_argument("--save_plot", default=True)  # Whether load offline model
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic

    parser.add_argument("--num_critics", default=48, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.num_critics}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("../results"):
        os.makedirs("../results")

    if args.save_model and not os.path.exists("../models"):
        os.makedirs("../models")

    # Create CSV file for logging
    csv_file = f"../results/{file_name}.csv"
    csv_initialized = False
    fieldnames = ['epoch']

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    policy = EMDSAC.EMDSAC(state_dim, action_dim, min_action, max_action, args.num_critics,args.device,args.save_plot,)

    ds = d4rl.qlearning_dataset(env)
    mean = np.mean(ds["observations"], 0)
    std = np.std(ds["observations"], 0) + 1e-3
    dataset_size = ds["observations"].shape[0]
    offline_replay_buffer = EnvReplayBuffer(int(1e6), env)
    for i in range(dataset_size):
        obs = (ds["observations"][i] - mean) / std
        new_obs = (ds["next_observations"][i] - mean) / std
        action = ds["actions"][i]
        reward = ds["rewards"][i]
        done = ds["terminals"][i]
        offline_replay_buffer.add_sample(obs, action, reward, done, new_obs)

    evaluations = []
    best_reward = 0

    for i in range(int(args.max_timesteps)+1):
        # Train agent after collecting sufficient data
        if i >= 0:
            train_data_offline = offline_replay_buffer.random_batch(args.batch_size)

            train_data = dict()
            train_data["obs"] = torch.FloatTensor(train_data_offline["observations"]).to(f'cuda:{args.device}')
            train_data["obs2"] = torch.FloatTensor(train_data_offline["next_observations"]).to(f'cuda:{args.device}')
            train_data["act"] = torch.FloatTensor(train_data_offline["actions"]).to(f'cuda:{args.device}')
            train_data["rew"] = torch.FloatTensor(train_data_offline["rewards"]).to(f'cuda:{args.device}').squeeze(1)
            train_data["done"] = torch.FloatTensor(train_data_offline["terminals"]).to(f'cuda:{args.device}').squeeze(1)

            info = policy.local_update(train_data)

            # Prepare data for CSV logging
            log_data = {'epoch': i}
            log_data.update(info)



        if i % args.eval_freq == 0:
            avg_reward, d4rl_score = eval_policy(policy, args.env, mean=mean, std=std)
            evaluations.append((avg_reward, d4rl_score))

            # Initialize CSV file with headers if not done
            if not csv_initialized:
                fieldnames.extend(list(info.keys()))
                fieldnames.extend(['avg_reward', 'd4rl_score'])  # Add evaluation columns

                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                csv_initialized = True

            # Write training info to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_data)

            # Update CSV with evaluation results
            try:
                # Read all rows
                rows = []
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                # Find the row for current epoch and add evaluation results
                for row in rows:
                    if int(row['epoch']) == i:
                        row['avg_reward'] = avg_reward
                        row['d4rl_score'] = d4rl_score
                        break

                # Write back all rows
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception as e:
                print(f"Error updating CSV with evaluation results: {e}")

            if evaluations[-1][0] > best_reward:
                best_reward = evaluations[-1][0]
                if args.save_model:
                    policy.save_checkpoint(f"../models/{file_name}_best.pth")
            print(f"Best reward: {best_reward: .3f}")

            print("---------------------------------------")
            np.save(f"../results/{file_name}", evaluations)
            if args.save_model:
                policy.save_checkpoint(f"../models/{file_name}.pth")
