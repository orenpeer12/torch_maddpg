import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch.nn as nn
import numpy as np

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, device="cuda:0", comm_size=0, comm=False, symbolic_comm=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.device = device
        # self.device = "cpu"
        self.comm = comm
        self.comm_size = comm_size
        self.symbolic_comm = symbolic_comm
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action).to(self.device)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False).to(self.device)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action).to(self.device)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False).to(self.device)

        if self.device == "cuda:0":
            self.policy = nn.DataParallel(self.policy)
            self.critic = nn.DataParallel(self.critic)
            self.target_policy = nn.DataParallel(self.target_policy)
            self.target_critic = nn.DataParallel(self.target_critic)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs).to(self.device)
        if self.discrete_action:
            if explore:
                if self.comm:
                    action = torch.cat([gumbel_softmax(action[:, :-self.comm_size], hard=True),
                                        gumbel_softmax(action[:, self.comm_size + 1:], hard=True)], axis=1)
                else:
                    action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                if self.comm:
                    a=1
                #     action[:, :-self.comm_size] += Variable(Tensor(self.exploration.noise()),
                #                        requires_grad=False)[:2]
                # else:
                action += Variable(Tensor(self.exploration.noise()), requires_grad=False)
            # action[:, :-self.comm_size] = action[:, :-self.comm_size].clamp(-1, 1)
        return action.clamp(-1, 1)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class Prey_Controller(object):
    def __init__(self, controller_radius=0.01, discrete_action=True, num_predators=3, num_obstacles=2):
        self.controller_radius = controller_radius
        self.num_predators = num_predators
        self.num_obstacles = num_obstacles
        self.discrete_action = discrete_action
        self.policy = self.target_policy = self.step
        # just to specify it has no learning capabilities...
        self.controller = True
        return

    def reset_noise(self):
        return

    def scale_noise(self, scale):
        return

    def step(self, obs, explore=False):
        # thin_obs = obs.reshape((obs.shape[-1]))
        # self_loc = obs[:, 2:4]
        predators_loc = obs[:, 4 + self.num_obstacles*2: 4 + self.num_obstacles*2 + self.num_predators*2]
        predators_loc = predators_loc.reshape((obs.shape[0], self.num_predators, 2))
        direction_away_from_predatores = - predators_loc
        dists = torch.norm(direction_away_from_predatores, dim=2).unsqueeze(-1)
        # nearby_idx = (dists < self.controller_radius)

        if not self.discrete_action:
            return (direction_away_from_predatores / (dists + 0.01)).mean(dim=1).clamp(-1, 1)
            # return direction_away_from_predatores.mean(dim=1).clamp(-1, 1)

            # Next code is not working and tries to run away from nearby predatores...
            # action = 2 * torch.rand_like(self_loc) - 1
            #
            # if nearby_idx.shape[0] is not 0:
            #     nearby_loc = predators_loc[nearby_idx[:, 0], nearby_idx[:, 1]]
            #     act_dir = (self_loc[nearby_idx[0][0], :] - nearby_loc).mean(dim=-1)
            #     act_dir /= torch.norm(act_dir, dim=-1)
            #     action[nearby_idx[:, 0]] = act_dir
            # return action

        elif self.discrete_action:
            if nearby_idx.shape[0] is 0:
                action = torch.zeros(size=(obs.shape[0], 5), requires_grad=False)
                action[0, torch.randint(low=0, high=5, size=(1,))] = 1
                return action
            else:
                pass    # TODO: add support to descrete actions!!

    # def step(self, obs, explore=False): # SLOW!
    #     if type(obs) is np.ndarray:
    #         obs = torch.Tensor(obs)
    #     thin_obs = obs.reshape((obs.shape[-1]))
    #     self_loc = thin_obs[:,2:4]
    #     predators_loc = thin_obs[4 + self.num_obstacles*2: 4 + self.num_obstacles*2 + self.num_predators*2]
    #     predators_loc = predators_loc.reshape((self.num_predators, 2))
    #     dists = torch.norm(self_loc - predators_loc, dim=1)
    #     nearby_idx = (dists < self.escape_radius).nonzero()[:,0]
    #     if nearby_idx.shape[0] is 0:
    #         if self.discrete_action:
    #             action = torch.zeros(size=(1,5), requires_grad=False)
    #             action[0, torch.randint(low=0, high=5, size=(1,))] = 1
    #             return action
    #         else:
    #             return 2 * torch.rand((1, 2)) - 1
    #     else:
    #         act_dir = torch.zeros((1, 2))
    #         nearby_loc = predators_loc[nearby_idx, :]
    #         act_dir[0, :] = (self_loc - nearby_loc).mean(dim=0)
    #         act_dir /= torch.norm(act_dir)
    #         if not self.discrete_action:
    #             return act_dir
    #         else:
    #             # TODO: add support to descrete actions!!
    #             pass
    #             # if action[0] > 0:
    #             #     if action[1] >0:
    #             #         return
    #     return action

    def get_params(self):
        return

    def load_params(self, params):
        return
