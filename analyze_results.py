import numpy as np
import matplotlib
import sys, time
if not sys.platform.startswith('win'):
    matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import json
import os, shutil
import torch
import time
import numpy as np
from torch_utils.make_env import make_parallel_env
from algorithms.maddpg import MADDPG
from torch_args import Arglist
from torch.autograd import Variable
import gc

if not sys.platform.startswith('win'):
    # from server
    base_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/test_model_comm/"
    path_to_summary = base_path + "logs/summary.json"
    path_to_rewards = base_path + "episodes_rewards.npy"

else:
    # from local
    base_path = "C:\\git\\torch_maddpg\\models\\simple_tag\\test_model_no_shape\\"
    path_to_summary = base_path + "logs\\summary.json"
    path_to_rewards = base_path + "run6\\episodes_rewards.npy"


CLEANUP = False
DISPLAY_LOSS = False
DISPLAY_SINGLE_RUN_REWARDS = False
DISPLAY_MEAN_RUN_REWARDS = False
SHOW_RUN = True

num_agents = 5

if DISPLAY_LOSS:
    # show loss funcs:
    with open(path_to_summary) as json_file:
        data = json.load(json_file)
    # data.keys()
    for loss in data.keys():
        plt.figure(loss)
        plt.plot([ep_loss[2] for ep_loss in data[loss]])
        plt.show()

if DISPLAY_SINGLE_RUN_REWARDS:
    rewards_data = np.load(path_to_rewards, allow_pickle=True).item()
    tot_ep_rewards = np.vstack(rewards_data["tot_ep_rewards"])
    mean_ep_rewards = np.vstack(rewards_data["mean_ep_rewards"])

    for agent in range(num_agents):
        agent_tot_ep_rewards = tot_ep_rewards[:, agent]
        agent_mean_ep_rewards = mean_ep_rewards[:, agent]
        plt.figure("agent " + str(agent))
        plt.subplot(2,1,1)
        plt.plot(agent_tot_ep_rewards)
        plt.title("total episodic reward")

        plt.subplot(2,1,2)
        plt.plot(agent_mean_ep_rewards)
        plt.title("mean episodic reward")
        plt.show()

if DISPLAY_MEAN_RUN_REWARDS:
    mean_reward_per_episode = [] # the mean reward of each agent each episode
    tot_reward_per_episode = [] # the tot cumulative reward of each agent each episode.
    runs = os.listdir(base_path)
    runs = [run for run in runs if run.startswith('run')]
    num_runs = len(runs)
    first = True
    for run in runs:
        gc.collect()
        rewards_data = np.load(base_path + run + "\\episodes_rewards.npy", allow_pickle=True).item()
        tot_ep_rewards = np.vstack(rewards_data["tot_ep_rewards"])
        mean_ep_rewards = np.vstack(rewards_data["mean_ep_rewards"])
        if first:
            num_episodes = len(mean_ep_rewards)
            mean_reward_per_episode = np.zeros((num_episodes, 4))
            tot_reward_per_episode = np.zeros((num_episodes, 4))
            first = False

        tot_reward_per_episode += tot_ep_rewards
        mean_reward_per_episode += mean_ep_rewards
    tot_reward_per_episode /= num_runs
    mean_reward_per_episode /= num_runs

    gc.collect()
    for agent, agent_type in zip([0, 3], ["Predators", "Prey"]):
        agent_tot_ep_rewards = tot_reward_per_episode[:, agent]
        agent_mean_ep_rewards = mean_reward_per_episode[:, agent]
        tot_ep_rew_mean_across_optim_step = []
        mean_ep_rew_mean_across_optim_step = []
        for i in range(0, 14999, 4):
            tot_ep_rew_mean_across_optim_step.append(agent_tot_ep_rewards[i:i + 4].mean())
            mean_ep_rew_mean_across_optim_step.append(agent_mean_ep_rewards[i:i + 4].mean())
        plt.figure("agent " + str(agent))
        plt.subplot(2,1,1)
        plt.plot(tot_ep_rew_mean_across_optim_step)
        plt.title(agent_type + " total episodic reward average across runs")
        plt.xlabel("Optimizer steps")
        plt.subplot(2,1,2)
        plt.plot(mean_ep_rew_mean_across_optim_step)
        plt.title(agent_type + " mean episodic reward average across runs")
    plt.show()

if SHOW_RUN:
    config = Arglist()
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.discrete_action)
    maddpg = MADDPG.init_from_save(config.load_model_path)
    # show some examples:
    for ep_i in range(0, 3):
        print("showing example number " + str(ep_i))
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')
        for et_i in range(100):
            env.env._render("human", False)
            time.sleep(0.1)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            obs = next_obs
        env.close()
#
#
#
#
# cleanup all intermediate saves
if CLEANUP:
    sub_folders = os.listdir(base_path)
    for sub_folder in sub_folders:
        if sub_folder.startswith('run'):
            if os.path.exists(base_path + sub_folder + "\\incremental"):
                shutil.rmtree(base_path + sub_folder + "\\incremental")
