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

# 01/12/19 14:32
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

        config.save(run_dir)
        if run_num is 0: config.print_args()
        logger = SummaryWriter(str(log_dir))

        if not config.USE_CUDA:
            torch.set_num_threads(config.n_training_threads)
        env = make_parallel_env(config)
        # add comm to action space:
        for a_i, a_type in enumerate(env.agent_types):
            if a_type is "adversary":
                env.action_space[a_i] = \
                    {'act': env.action_space[a_i], 'comm': Discrete(config.predators_comm_size)}
            else:
                env.action_space[a_i] =\
                    {'act': env.action_space[a_i], 'comm': Discrete(0)}

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
        for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
            ep_rewards = np.zeros((1, len(env.agent_types)))
            if ep_i % 100 == 0:
                printProgressBar(ep_i, start_time, config.n_episodes, "run" + str(run_num) + ": Episodes Done: ", "", 20, "%")
                # print("Episodes %i-%i of %i" % (ep_i + 1,
                #                                 ep_i + 1 + config.n_rollout_threads,
                #                                 config.n_episodes))
            obs = env.reset()
            # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
            maddpg.prep_rollouts(device=config.device)

            explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
            maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
            maddpg.reset_noise()

            for et_i in range(config.episode_length):
                # if ep_i > 1000:
                #     env.env._render("human", False)
                #     time.sleep(0.1)
                # rearrange observations to be per agent, and convert to torch Variable
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
                ep_rewards += rewards
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs
                t += config.n_rollout_threads
                if (len(replay_buffer) >= config.batch_size and
                        (t % config.steps_per_update) < config.n_rollout_threads):

                    maddpg.prep_training(device=config.device)

                    for u_i in range(config.n_rollout_threads):
                        for a_i in range(maddpg.nagents):
                            if maddpg.alg_types[a_i] is 'CONTROLLER':
                                continue
                            sample = replay_buffer.sample(config.batch_size,
                                                          to_gpu=config.USE_CUDA)
                            maddpg.update(sample, a_i, logger=logger)
                        maddpg.update_all_targets()
                    maddpg.prep_rollouts(device=config.device)

            # ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
            mean_ep_rewards.append(ep_rewards / config.episode_length)
            all_ep_rewards.append(ep_rewards)
            if ep_i == config.n_episodes - 1:
                printProgressBar(ep_i, start_time, config.n_episodes, "run" + str(run_num) + ": Episodes Done ", "", 20, "%")

            # for a_i, a_ep_rew in enumerate(ep_rews):
            #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

            # if ep_i % config.save_interval < config.n_rollout_threads:
            #     os.makedirs(run_dir / 'incremental', exist_ok=True)
            #     maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            #     maddpg.save(run_dir / 'model.pt')

        np.save(run_dir / 'episodes_rewards', {"tot_ep_rewards": all_ep_rewards.copy(),
                                               "mean_ep_rewards": mean_ep_rewards.copy()}, True)
        maddpg.save(run_dir / 'model.pt')
        # env.close()
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()

