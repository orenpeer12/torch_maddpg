import argparse
import torch
import time
import os, sys
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch_utils.make_env import make_parallel_env
from torch_utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG
from torch_args import Arglist

do_log = False
MAKE_NEW_LOG = True
LOAD_MODEL = False
# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
num_runs = 100

if USE_CUDA:
    device = "cuda:0"
else:
    device = "cpu"


if __name__ == '__main__':
    if not LOAD_MODEL:
        for i in range(num_runs):
            config = Arglist()
            model_dir = Path('./models') / config.env_id / config.model_name
            if not model_dir.exists():
                curr_run = 'run1'
            else:
                if MAKE_NEW_LOG:
                    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                                     model_dir.iterdir() if
                                     str(folder.name).startswith('run')]
                    if len(exst_run_nums) == 0:
                        curr_run = 'run1'
                    else:
                        curr_run = 'run%i' % (max(exst_run_nums) + 1)
                else:
                    curr_run = 'last_run_override'
            run_dir = model_dir / curr_run
            log_dir = run_dir / 'logs'
            if not log_dir.exists():
                os.makedirs(log_dir)
            logger = SummaryWriter(str(log_dir))

            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            if not USE_CUDA:
                torch.set_num_threads(config.n_training_threads)
            env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                                    config.discrete_action)
            maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                          adversary_alg=config.adversary_alg,
                                          tau=config.tau,
                                          lr=config.lr,
                                          hidden_dim=config.hidden_dim, device=device)
            replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                         [obsp.shape[0] for obsp in env.observation_space],
                                         [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                          for acsp in env.action_space])
            t = 0
            all_ep_rewards = []
            mean_ep_rewards = []
            for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
                ep_rewards = np.zeros((1, len(env.agent_types)))
                print("Episodes %i-%i of %i" % (ep_i + 1,
                                                ep_i + 1 + config.n_rollout_threads,
                                                config.n_episodes))
                obs = env.reset()
                # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
                maddpg.prep_rollouts(device=device)

                explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
                maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
                maddpg.reset_noise()

                for et_i in range(config.episode_length):
                    # rearrange observations to be per agent, and convert to torch Variable
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).to(device),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]
                    # get actions as torch Variables
                    torch_agent_actions = maddpg.step(torch_obs, explore=True)
                    # convert actions to numpy arrays
                    agent_actions = [ac.detach().cpu().data.numpy() for ac in torch_agent_actions]
                    # rearrange actions to be per environment
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    ep_rewards += rewards
                    replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                    obs = next_obs
                    t += config.n_rollout_threads
                    if (len(replay_buffer) >= config.batch_size and
                        (t % config.steps_per_update) < config.n_rollout_threads):

                        maddpg.prep_training(device=device)

                        for u_i in range(config.n_rollout_threads):
                            for a_i in range(maddpg.nagents):
                                sample = replay_buffer.sample(config.batch_size,
                                                              to_gpu=USE_CUDA)
                                maddpg.update(sample, a_i, logger=logger)
                            maddpg.update_all_targets()
                        maddpg.prep_rollouts(device=device)
                ep_rews = replay_buffer.get_average_rewards(
                    config.episode_length * config.n_rollout_threads)

                mean_ep_rewards.append(ep_rewards / config.episode_length)
                all_ep_rewards.append(ep_rewards)

                for a_i, a_ep_rew in enumerate(ep_rews):
                    logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

                if ep_i % config.save_interval < config.n_rollout_threads:
                    os.makedirs(run_dir / 'incremental', exist_ok=True)
                    maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
                    maddpg.save(run_dir / 'model.pt')

            np.save(run_dir / 'episodes_rewards', {"tot_ep_rewards": all_ep_rewards, "mean_ep_rewards": mean_ep_rewards}, True)
            maddpg.save(run_dir / 'model.pt')
            # env.close()
            logger.export_scalars_to_json(str(log_dir / 'summary.json'))
            logger.close()

    else:
        config = Arglist()
        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                                config.discrete_action)
        maddpg = MADDPG.init_from_save(config.load_model_path)
        # show some examples:
        for ep_i in range(0, 3):
            print("showing example number " + str(ep_i))
            obs = env.reset()
            maddpg.prep_rollouts(device=device)
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