import torch
import time
import os, sys, gc
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.printer import *
from utils.make_env import make_parallel_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG
from torch_args import Arglist
from utils.maddpg_utils import *

# 07/12/19 18:32
do_log = False
MAKE_NEW_LOG = True
LOAD_MODEL = False


if __name__ == '__main__':
    config = Arglist()
    num_runs = config.num_runs

    for run_num in range(num_runs):
        gc.collect()
        model_dir = Path('./models') / config.env_id / config.model_name
        if not model_dir.exists():
            curr_run = 'run0'
        else:
            if MAKE_NEW_LOG:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                                 model_dir.iterdir() if
                                 str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run0'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            else:
                curr_run = 'last_run_override'
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        if not log_dir.exists():
            os.makedirs(log_dir)

        config.save_args(run_dir)
        if run_num is 0: config.print_args()
        logger = SummaryWriter(str(log_dir))

        if not config.USE_CUDA:
            torch.set_num_threads(config.n_training_threads)
        env = make_parallel_env(config)
        eval_env = make_parallel_env(config)

        # add comm to action space:
        for a_i, a_type in enumerate(env.agent_types):
            if a_type is "adversary":
                env.action_space[a_i] = \
                    {'act': env.action_space[a_i], 'comm': Discrete(config.predators_comm_size)}
                eval_env.action_space[a_i] = \
                    {'act': eval_env.action_space[a_i], 'comm': Discrete(config.predators_comm_size)}
            else:
                env.action_space[a_i] =\
                    {'act': env.action_space[a_i], 'comm': Discrete(0)}
                eval_env.action_space[a_i] = \
                    {'act': eval_env.action_space[a_i], 'comm': Discrete(0)}

        maddpg = MADDPG.init_from_env(env, config)
        replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp['comm'].n + acsp['act'].n if config.discrete_action else acsp['comm'].n + acsp['act'].shape[0]
                                      for acsp in env.action_space])
                                     # [acsp['comm'].n + acsp['act'].n if isinstance(acsp, dict) else acsp.n
                                     #  for acsp in env.action_space])
        t = 0
        # reset test results arrays
        all_ep_rewards = []
        mean_ep_rewards = []
        start_time = time.time()
        step = 0
        win_counter = 0
        curr_ep = -1
        eval_win_rates = [0]

        while step < config.n_time_steps:   # total steps to be performed during a single run
            # start a episode due to episode termination\done
            curr_ep += 1
            ep_rewards = np.zeros((1, len(env.agent_types)))    # init reward vec for single episode.

            # prepare episodic stuff
            obs = env.reset()
            maddpg.prep_rollouts(device=config.device)
            explr_pct_remaining = max(0, config.n_exploration_steps - step) / config.n_exploration_steps
            maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
            maddpg.reset_noise()

            for ep_step in range(config.episode_length):    # 1 episode loop. ends due to term\done
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, ind])),
                                      requires_grad=False)
                             for ind in range(maddpg.nagents)]
                # get actions as torch Variables
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
                # convert actions to numpy arrays
                # agent_actions = [ac.detach().cpu().data.numpy() for ac in torch_agent_actions]
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[idx] for ac in agent_actions] for idx in range(config.n_rollout_threads)]
                next_obs, rewards, dones, infos = env.step(actions)

                if (len(replay_buffer) >= config.batch_size and
                        (step % config.steps_per_eval) < config.n_rollout_threads):
                    eval_win_rates.append(eval_model(maddpg, eval_env, config.episode_length, config.num_steps_in_eval,
                                                     config.n_rollout_threads))

                if (len(replay_buffer) >= config.batch_size and
                        (step % config.steps_per_update) < config.n_rollout_threads):
                    train_model(maddpg, config, replay_buffer)


                step += config.n_rollout_threads  # advance the step-counter
                ep_rewards += rewards
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

                if dones.any(): # terminate episode if won! #
                    win_counter += 1
                    break
                obs = next_obs

            # ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
            mean_ep_rewards.append(ep_rewards / config.episode_length)
            all_ep_rewards.append(ep_rewards)

            if step % 100 == 0 or (step == config.n_time_steps - 1):    # print progress.
                printProgressBar(step, start_time, config.n_time_steps, "run" + str(run_num) + ": Steps Done: ",
                                 " Last eval win rate: {0:.2%}".format(eval_win_rates[-1]), 20, "%")

            # for a_i, a_ep_rew in enumerate(ep_rews):
            #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

            # if ep_i % config.save_interval < config.n_rollout_threads:
            #     os.makedirs(run_dir / 'incremental', exist_ok=True)
            #     maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            #     maddpg.save(run_dir / 'model.pt')

        np.save(run_dir / 'episodes_rewards', {"tot_ep_rewards": all_ep_rewards.copy(),
                                               "mean_ep_rewards": mean_ep_rewards.copy()}, True)
        np.save(run_dir / 'win_rates', eval_win_rates, True)
        maddpg.save(run_dir / 'model.pt')
        # env.close()
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()

