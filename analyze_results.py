import numpy as np
import matplotlib
import sys
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
from gym.spaces import Discrete
from pathlib import Path


if not sys.platform.startswith('win'):
    # from server
    base_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/"
    path_to_summary = base_path + "logs/summary.json"

else:
    # from local
    base_path = Path("C:\\git\\results_predators\\")
    base_path = Path("C:\\git\\torch_maddpg\\models\\simple_tag\\")
    # base_path = "C:\\git\\torch_maddpg\\results_predators\\test_model_max_not_min"
    path_to_summary = base_path / "logs\\summary.json"


CLEANUP = False
DISPLAY_LOSS = False
DISPLAY_SINGLE_RUN_REWARDS = False
DISPLAY_MEAN_RUN_REWARDS = True
SHOW_RUN = True

# models_to_compare = \
    # ['1prey_thin_obs_space', '1prey_thinObs_critic_comm_noShaping', '1prey_thinObs_critic_comm_withShaping',
    #  "1prey_thinObs_critic_comm_withShaping_cont_act"]
models_to_compare = ["2prey_thin_obs_space", "2prey_thinObs_critic_comm_withShaping", "2prey_thinObs_critic_comm_withShaping_cont_act"]
# models_to_compare = ["test_model_baseline_1prey", "1prey_thin_obs_space"]
models_to_compare = ["1prey_contAct_thinObs_noCom"]

num_agents = 4

if DISPLAY_LOSS:
    # show loss funcs:
    with open(path_to_summary) as json_file:
        data = json.load(json_file)
    # data.keys()
    for loss in data.keys():
        plt.figure(loss)
        plt.plot([ep_loss[2] for ep_loss in data[loss]])
        plt.show()

if DISPLAY_MEAN_RUN_REWARDS:
    plt.figure("Predators")
    plt.title("Predators total episodic reward average across runs")
    plt.xlabel("Optimizer steps")
    plt.figure("Prey")
    plt.title("Prey total episodic reward average across runs")
    plt.xlabel("Optimizer steps")

    for model in models_to_compare:
        mean_reward_per_episode = []    # the mean reward of each agent each episode
        tot_reward_per_episode = []     # the tot cumulative reward of each agent each episode.
        runs = os.listdir(base_path / model)
        # runs = [run for run in runs if run.startswith('run')]
        runs = [[run for run in runs if run.startswith('run')][0]]
        num_runs = len(runs)
        first = True
        for run in runs:
            gc.collect()
            rewards_data = np.load(base_path /model / run / "episodes_rewards.npy", allow_pickle=True).item()
            tot_ep_rewards = np.vstack(rewards_data["tot_ep_rewards"])
            mean_ep_rewards = np.vstack(rewards_data["mean_ep_rewards"])
            if first:
                num_episodes = len(mean_ep_rewards)
                mean_reward_per_episode = np.zeros((num_episodes, num_agents))
                tot_reward_per_episode = np.zeros((num_episodes, num_agents))
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
            plt.figure(agent_type)
            plt.plot(tot_ep_rew_mean_across_optim_step)

            # plt.plot(mean_ep_rew_mean_across_optim_step)
    for model in ["Predators", "Prey"]:
        plt.figure(model)
        plt.legend(models_to_compare)
    plt.show()

if SHOW_RUN:
    config = Arglist()
    env = make_parallel_env(config)
    # add comm to action space:
    for i, a_type in enumerate(env.agent_types):
        if a_type is "adversary":
            env.action_space[i] = \
                {'act': env.action_space[i], 'comm': Discrete(config.predators_comm_size)}
        else:
            env.action_space[i] = \
                {'act': env.action_space[i], 'comm': Discrete(0)}

    maddpg = MADDPG.init_from_save(base_path /"2prey_contAct_thinObs_noCom" / "run0\model.pt" )
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
