import numpy as np
import torch
from torch.autograd import Variable
import time

def train_model(maddpg, config, replay_buffer):
    maddpg.prep_training(device=config.device)
    for u_i in range(config.n_rollout_threads):
        for a_i in range(maddpg.nagents):
            if maddpg.alg_types[a_i] is 'CONTROLLER':
                continue
            sample = replay_buffer.sample(config.batch_size,
                                          to_gpu=config.USE_CUDA)
            maddpg.update(sample, a_i)
        maddpg.update_all_targets()
    maddpg.prep_rollouts(device=config.device)


def eval_model(maddpg, eval_env, ep_len, num_steps, rollout_threads, display=False):
    eval_step = -1
    eval_wins = 0
    num_episodes = -1
    while eval_step < num_steps:  # total steps to be performed during a single eval
        num_episodes += 1
        eval_obs = eval_env.reset()
        for eval_ep_step in range(ep_len):
            if display:
                eval_env.env._render("human", False)
                time.sleep(0.05)
            eval_step += 1
            eval_torch_obs = [Variable(torch.Tensor(np.vstack(eval_obs[:, ind])), requires_grad=False)
                              for ind in range(maddpg.nagents)]
            eval_torch_agent_actions = maddpg.step(eval_torch_obs, explore=True)
            eval_agent_actions = [ac.cpu().data.numpy() for ac in eval_torch_agent_actions]
            eval_actions = [[ac[idx] for ac in eval_agent_actions] for idx in range(rollout_threads)]
            eval_next_obs, eval_rewards, eval_dones, eval_infos = eval_env.step(eval_actions)
            if eval_dones.any():  # terminate episode if won!
                eval_wins += 1
                break
            eval_obs = eval_next_obs
    win_rate = eval_wins / num_episodes
    return win_rate
