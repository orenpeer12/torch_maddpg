import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, config):
        self.shaping = config.shaping
        self.predators_comm = config.predators_comm
        world = World()
        # set any world properties first
        world.dim_c = config.predators_comm_size if config.predators_comm else 0
        num_prey = config.num_prey
        num_predators = config.num_predators
        num_agents = num_predators + num_prey
        num_landmarks = config.num_landmarks
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.adversary = True if i < num_predators else False
            # if config.predators_comm:
            #     agent.silent = False if agent.adversary else True   # Oren
            # else:
            #     agent.silent = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            # agent.accel = 3.0 if agent.adversary else 2.0
            # agent.max_speed = 1.0 if agent.adversary else 0.5

            agent.accel = config.pred_acc if agent.adversary else config.prey_acc
            agent.max_speed = config.pred_max_speed if agent.adversary else config.prey_max_speed

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = \
                np.array([0.35 + 0.2 * int(agent.name[-1]), 0.85 - 0.3 * int(agent.name[-1]), 0.35 ]) if not agent.adversary else np.array([0.85 - 0.3 * int( agent.name[-1]), 0.35, 0.35 + 0.3 * int(agent.name[-1])])
            # print(agent.color)
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = self.shaping
        # print(self.shaping)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = self.shaping
        # print(self.shaping)
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * sum([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
                # rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                # if ag.name == "agent 1":  ###
                #     continue        ###
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                # entity_pos.append(np.array([11, 11]))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            if other.adversary:
                comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # other_pos.append(np.array([22, 22]))
            if not other.adversary:
                other_vel.append(other.state.p_vel)
                # other_vel.append(np.array([33, 33]))
        if self.predators_comm:
            if agent.adversary:
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos +
                                  other_vel + comm)     # OREN
            else:
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
            # return np.concatenate([np.array([np.int(agent.name[-1]), np.int(agent.name[-1])])] +
            #                       [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def is_done(self, agent, world):
        if agent in self.good_agents(world):
            return False

        for prey in self.good_agents(world):
            if self.is_collision(agent, prey):
                return True
        return False

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     for other in world.agents:
    #         if other is agent:
    #             continue
    #         if other.adversary:
    #             comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         if not other.adversary:
    #             other_vel.append(other.state.p_vel)
    #     # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + comm) # OREN
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
