import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent, Prey_Controller
import numpy as np
MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, group_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, device='cuda:0',
                 discrete_action=False, predators_comm=False, predators_comm_size=0, symbolic_comm=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.predators_comm = predators_comm
        self.predators_comm_size = predators_comm_size
        self.symbolic_comm = symbolic_comm
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = []
        for ag_i, params in enumerate(agent_init_params):
            if alg_types[ag_i] in ["MADDPG", "DDPG"]:
                self.agents.append(DDPGAgent(ag_id=ag_i, lr=lr, discrete_action=discrete_action,
                                             hidden_dim=hidden_dim, device=device,
                                             comm=predators_comm if alg_types[ag_i] is 'MADDPG' else False,
                                             comm_size=predators_comm_size, group_type=group_types[ag_i], **params))
            elif alg_types[ag_i] == "CONTROLLER":
                self.agents.append(Prey_Controller(**params))
        # for agent in self.agents:
        #     print("An agent of type: ")
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        # self.pol_dev = 'cpu'  # device for policies
        # self.critic_dev = 'cpu'  # device for critics
        # self.trgt_pol_dev = 'cpu'  # device for target policies
        # self.trgt_critic_dev = 'cpu'  # device for target critics
        self.pol_dev = device  # device for policies
        self.critic_dev = device  # device for critics
        self.trgt_pol_dev = device  # device for target policies
        self.trgt_critic_dev = device  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def take_only_new_info(self, agent_idx, some_obs):
        new_info_obs = [other_obs[:, 0:2] for i, other_obs in enumerate(some_obs) if
                        i is not agent_idx and self.agents[i].group_type is 'adversary']
        # add them to the current agent observation
        new_info_obs.append(some_obs[agent_idx])
        return new_info_obs

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action:    # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                # all_trgt_acs = []
                # for pi, nobs, alg_type in zip(self.target_policies, next_obs, self.alg_types):
                #     if alg_type in ['MADDPG', 'DDPG']:
                #         all_trgt_acs.append(pi(nobs))
                #     else:
                #         all_trgt_acs.append(pi(nobs))
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies, next_obs)]
            # trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) # ORIGINAL

            # take only the speeds of the *other* MADDPG agents.
            new_info_next_obs = self.take_only_new_info(agent_i, next_obs)
            trgt_vf_in = torch.cat((*new_info_next_obs, *all_trgt_acs), dim=1)  # Oren - take only NEW information from other agents (i.e. their speeds)

        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            # vf_in = torch.cat((*obs, *acs), dim=1) # original
            # take %only% the speeds of the *other* MADDPG agents.
            new_info_obs = self.take_only_new_info(agent_i, obs)
            vf_in = torch.cat((*new_info_obs, *acs), dim=1)     # Oren - take only NEW information from other agents (i.e. their speeds)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
            ##

        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob, a_type in zip(range(self.nagents), self.policies, obs, self.alg_types):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))



            new_info_obs = self.take_only_new_info(agent_i, obs)
            vf_in = torch.cat((*new_info_obs, *all_pol_acs), dim=1)
            # vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            if hasattr(a, "controller"):
                continue
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device="cuda:0"):
        for a in self.agents:
            if hasattr(a, "controller"):
                continue
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == "cuda:0":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            if isinstance(a, DDPGAgent):
                a.policy.eval()
        if device == "cuda:0":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, config):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_alg = config.agent_alg
        adversary_alg = config.adversary_alg
        gamma = config.gamma
        tau = config.tau
        lr = config.lr
        hidden_dim = config.hidden_dim
        device = config.device

        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        group_types = env.agent_types
        # for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
        #                                alg_types)): # ORIG
        for acsp, obsp, algtype, curr_agent in zip(env.action_space, env.observation_space,
                                       alg_types, range(len(alg_types))):

            num_in_pol = obsp.shape[0]
            if isinstance(acsp['act'], Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
                num_out_pol = acsp['comm'].n + acsp['act'].shape[0]

            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
                num_out_pol = acsp['comm'].n + acsp['act'].n

            # if algtype == "MADDPG": # orig
            #     num_in_critic = 0
            #     for oobsp in env.observation_space:
            #         num_in_critic += oobsp.shape[0]
            #     for oacsp in env.action_space:
            #         num_in_critic += get_shape(oacsp)

            if group_types[curr_agent] == 'adversary':
                num_in_critic = env.observation_space[curr_agent].shape[0]
                # for oobsp in env.observation_space:
                #     num_in_critic += oobsp.shape[0]
                for i, oobsp in enumerate(env.observation_space):  # OREN - share only meaningful info.
                    if group_types[i] is 'adversary' and (i is not curr_agent):
                        num_in_critic += 2
                for oacsp in env.action_space:
                    num_in_critic += \
                        oacsp['comm'].n + oacsp['act'].n if config.discrete_action else oacsp['comm'].n + oacsp['act'].shape[0]

            # for oacsp in env.action_space:
                #     if isinstance(oacsp, dict):
                #         num_in_critic += oacsp['comm'].n + oacsp['act'].n
                    # else:
                    #     num_in_critic += get_shape(oacsp)
            elif algtype is "DDPG":
                num_in_critic = obsp.shape[0] + acsp['comm'].n
                num_in_critic += acsp['act'].n if config.discrete_action else acsp['act'].shape[0]

            elif algtype is "CONTROLLER":
                agent_init_params.append({"discrete_action": config.discrete_action,
                                          "num_predators": config.num_predators,
                                          "num_obstacles": config.num_landmarks})
                continue

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})

        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'group_types': group_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': config.discrete_action,
                     'device': device,
                     'predators_comm': config.predators_comm,
                     'predators_comm_size': config.predators_comm_size,
                     'symbolic_comm': config.symbolic_comm}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
