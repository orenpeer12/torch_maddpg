import sys
import json
import os, shutil
import torch
import time
import numpy as np
from utils.make_env import make_parallel_env
from algorithms.maddpg import MADDPG
from torch_args import Arglist
from torch.autograd import Variable
import gc
from gym.spaces import Discrete
from pathlib import Path
from utils.my_plotting import *
import numpy as np
import matplotlib
from utils.maddpg_utils import *

if not sys.platform.startswith('win'):
    matplotlib.use('tkagg')
import matplotlib.pyplot as plt

if not sys.platform.startswith('win'):
    # from server
    base_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/"
    path_to_summary = base_path + "logs/summary.json"

else:
    # from local
    # base_path = Path("C:\\git\\results_predators\\prey_controller\\baseline_winRate")
    base_path = Path("C:\\git\\results_predators\\baselines")
    # base_path = Path("C:\\git\\torch_maddpg\\models\\simple_tag\\")
    # base_path = Path("C:\\git\\torch_maddpg\\models\\simple_tag\\")
    path_to_summary = base_path / "logs\\summary.json"


CLEANUP = False
# DISPLAY_LOSS = False

DISPLAY_MEAN_RUN_REWARDS = False
SHOW_RUN = False
DISPLAY_MEAN_WIN_RATES = True
SMOOTH = True

models_to_compare = [#"1prey_1pred_0landmarks_noWalls_noCom_sumShape_noLand_noIL_controllerPray_SlowPrey",
                     #"1prey_1pred_0landmarks_noWalls_noCom_sumShape_noLand_noIL_DDPGpray_SlowPrey",
                     # "1prey_3pred_0landmarks_noWalls_noCom_sumShape_noLand_noIL_controllerPray_SlowPrey",
                     # "1prey_3pred_0landmarks_noWalls_noCom_sumShape_noLand_noIL_DDPGpray_SlowPrey",
                     # "2prey_1pred_0landmarks_noWalls_noCom_sumShape_noLand_noIL_controllerPray_SlowPrey",
                     # "2prey_1pred_0landmarks_noWalls_noCom_sumShape_noLand_withIL_controllerPray_SlowPrey",
                     "2prey_2pred_0landmarks_withWalls_withCom1_noShape_noLand_withIL_DDPGprey_SameSpeedPrey",
                     "2prey_2pred_0landmarks_withWalls_withCom1_noShape_noLand_noIL_DDPGprey_SameSpeedPrey",
                     "2prey_2pred_0landmarks_withWalls_noCom_noShape_noLand_withIL_DDPGprey_SameSpeedPrey"
]
# ]
num_agents = 5

if SHOW_RUN:
    cur_model = 5
    # see_runs = [ind for ind in range(6)]
    see_runs = [0]
    wait = 0.05
    ep_len = 50


    for cur_run in see_runs:
        for i in range(4):
            config = Arglist()
            config.load_args(base_path / models_to_compare[cur_model] / ("run" + str(cur_run)))
            env = make_parallel_env(config)
            model_path = base_path / models_to_compare[cur_model] / ("run" + str(cur_run)) / "model.pt"
            print(model_path)

            # add comm to action space:
            maddpg = MADDPG.init_from_save(model_path)
            # show some examples:
            obs = env.reset()
            # env.env._render("human", True)
            maddpg.prep_rollouts(device='cpu')

            # eval_model(maddpg, env, ep_len=100, num_steps=500, rollout_threads=1, display=True)
            for step in range(ep_len):
                env.env._render("human", False)
                time.sleep(wait)
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
        # env.env._render("human", True)
        # env.get_viewer().close()
        # env.close()


if DISPLAY_MEAN_RUN_REWARDS:
    plt.figure("Predators")
    plt.title("Predators total episodic reward average across runs")
    plt.xlabel("Optimizer steps")
    # plt.figure("Prey")
    # plt.title("Prey total episodic reward average across runs")
    # plt.xlabel("Optimizer steps")

    # for model, num_agents in zip(models_to_compare, [2]):
    for model, num_agents in zip(models_to_compare, [2,2,3,3]):
        mean_reward_per_episode = []    # the mean reward of each agent each episode
        tot_reward_per_episode = []     # the tot cumulative reward of each agent each episode.
        runs = os.listdir(base_path / model)
        runs = [[run for run in runs if run.startswith('run')][-1]]
        num_runs = len(runs)

        arguments = Arglist()
        # num_agents = arguments.num_predators + arguments.num_prey

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

        # for agent, agent_type in zip([0, 1], ["Predators", "Prey"]):
        agent = 0
        agent_type = "Predators"
        agent_tot_ep_rewards = tot_reward_per_episode[:, agent]
        agent_mean_ep_rewards = mean_reward_per_episode[:, agent]
        tot_ep_rew_mean_across_optim_step = []
        mean_ep_rew_mean_across_optim_step = []
        for i in range(0, agent_tot_ep_rewards.shape[0] - 1, 4):
            tot_ep_rew_mean_across_optim_step.append(agent_tot_ep_rewards[i:i + 4].mean())
            mean_ep_rew_mean_across_optim_step.append(agent_mean_ep_rewards[i:i + 4].mean())
        plt.figure(agent_type)
        if SMOOTH: tot_ep_rew_mean_across_optim_step = smooth(tot_ep_rew_mean_across_optim_step)
        plt.plot(tot_ep_rew_mean_across_optim_step, linewidth=1)
        ##
            # plt.plot(mean_ep_rew_mean_across_optim_step)
    # for model in ["Predators", "Prey"]:
    #     plt.figure(model)
    plt.legend([m.replace("_contAct_thinObs", "").replace("agent", "prey") for m in models_to_compare])
    plt.show()
    # set_default_mpl()

if DISPLAY_MEAN_WIN_RATES:
    plt.figure("Predators")
    plt.title("Predators total evaluation win-rates, averaged across runs")
    plt.xlabel("Evaluation #")
    # plt.figure("Prey")
    # plt.title("Prey total episodic reward average across runs")
    # plt.xlabel("Optimizer steps")

    # for model, num_agents in zip(models_to_compare, [2]):
    for model in models_to_compare:
        mean_reward_per_episode = []    # the mean reward of each agent each episode
        tot_reward_per_episode = []     # the tot cumulative reward of each agent each episode.
        runs = os.listdir(base_path / model)
        runs = [run for run in runs if run.startswith('run')]
        num_runs = len(runs)

        arguments = Arglist()
        # num_agents = arguments.num_predators + arguments.num_prey

        first = True
        for run in runs:
            arguments.load_args(base_path /model / run)
            num_agents = arguments.num_prey + arguments.num_predators
            gc.collect()
            win_rate = np.load(base_path /model / run / "win_rates.npy", allow_pickle=True)
            if first:
                win_rates = np.zeros(win_rate.shape)
                first = False
            ####
            if win_rate.shape[0] == 76:
                win_rate = win_rate[:-1]
            ####
            win_rates += win_rate

        win_rates /= num_runs

        gc.collect()

        # for agent, agent_type in zip([0, 1], ["Predators", "Prey"]):
        agent = 0
        agent_type = "Predators"

        plt.figure(agent_type)

        plt.plot(win_rates, linewidth=1)
        ##
            # plt.plot(mean_ep_rew_mean_across_optim_step)
    # for model in ["Predators", "Prey"]:
    #     plt.figure(model)
    plt.legend([m.replace("_contAct_thinObs", "").replace("agent", "prey") for m in models_to_compare])
    plt.show()
    set_default_mpl()
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

# if DISPLAY_LOSS:
#     # show loss funcs:
#     with open(path_to_summary) as json_file:
#         data = json.load(json_file)
#     # data.keys()
#     for loss in data.keys():
#         plt.figure(loss)
#         plt.plot([ep_loss[2] for ep_loss in data[loss]])
#         plt.show()
