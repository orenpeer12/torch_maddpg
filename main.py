import torch
import time
import os, sys, gc
import numpy as np
from utils.make_env import make_parallel_env
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG
from torch_args import Arglist
from utils.maddpg_utils import *
from utils.agents import IL_Controller
from utils.general_functions import *

# 27/12/19 10:20
do_log = False
MAKE_NEW_LOG = True
LOAD_MODEL = False
MODE = "RUN"    # "DEBUG"

if __name__ == '__main__':
    config = Arglist()
    num_runs = config.num_runs
    run_manager = running_env_manager(MODE)
    for run_num in range(num_runs):
        run_manager.prep_running_env(config, run_num)

        if not config.USE_CUDA:
            torch.set_num_threads(config.n_training_threads)
        env = make_parallel_env(config)
        eval_env = make_parallel_env(config)

        maddpg = MADDPG.init_from_env(env, config)
        IL_controller = IL_Controller(config)  # imitation learning controller
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
        # eps_without_IL = 0
        # eps_without_IL_hist = []

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
                        (step % config.steps_per_eval) < config.n_rollout_threads):     # perform evaluation
                    eval_win_rates.append(eval_model(maddpg, eval_env, config.episode_length, config.num_steps_in_eval,
                                                     config.n_rollout_threads, display=False))

                if (len(replay_buffer) >= config.batch_size and
                        (step % config.steps_per_update) < config.n_rollout_threads):   # perform training
                    train_model(maddpg, config, replay_buffer)

                step += config.n_rollout_threads  # advance the step-counter

                if (len(replay_buffer) >= config.batch_size and
                        (step % config.IL_inject_every) < config.n_rollout_threads):  # perform IL injection
                    step, eval_win_rates = \
                        IL_controller.IL_inject(maddpg, replay_buffer, eval_env, step, config, eval_win_rates)
                    IL_controller.decay()

                ep_rewards += rewards
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

                if dones.any(): # terminate episode if won! #
                    win_counter += 1
                    # eps_without_IL += 1
                    break
                obs = next_obs

            # perform IL injection if failed
            # if config.use_IL and ep_step == config.episode_length-1 and not dones.any():
            #     step, eval_win_rates = \
            #         IL_controller.IL_inject(maddpg, replay_buffer, eval_env, step, config, eval_win_rates)
            #     eps_without_IL_hist.append(eps_without_IL)
            #     eps_without_IL = 0

            mean_ep_rewards.append(ep_rewards / config.episode_length)
            all_ep_rewards.append(ep_rewards)

            if step % 100 == 0 or (step == config.n_time_steps):    # print progress.
                run_manager.printProgressBar(step, start_time, config.n_time_steps, "run" + str(run_num) + ": Steps Done: ",
                                 " Last eval win rate: {0:.2%}".format(eval_win_rates[-1]), 20, "%")

            # for a_i, a_ep_rew in enumerate(ep_rews):
            #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

            # if ep_i % config.save_interval < config.n_rollout_threads:
            #     os.makedirs(run_dir / 'incremental', exist_ok=True)
            #     maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            #     maddpg.save(run_dir / 'model.pt')

        # eps_without_IL_hist.append(eps_without_IL)
        if MODE == "RUN":
            run_dir = run_manager.run_dir
            np.save(run_dir / 'episodes_rewards', {"tot_ep_rewards": all_ep_rewards.copy(),
                                                   "mean_ep_rewards": mean_ep_rewards.copy()}, True)
            # np.save(run_dir / 'IL_hist', eps_without_IL_hist, True)
            np.save(run_dir / 'win_rates', eval_win_rates, True)
            maddpg.save(run_dir / 'model.pt')
        # env.close()
        # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        # logger.close()

