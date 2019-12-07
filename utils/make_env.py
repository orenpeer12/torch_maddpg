"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, config, benchmark=False, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(config)
    # create multiagent environment
    ####
    # def __init__(self, world, reset_callback=None, reward_callback=None,
    #              observation_callback=None, info_callback=None,
    #              done_callback=None, post_step_callback=None,
    #              shared_viewer=True, discrete_action=False, config=None)
    ####
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data, config=config,
                            done_callback=scenario.is_done, discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world=world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                            observation_callback=scenario.observation, config=config, done_callback=scenario.is_done,
                            discrete_action=discrete_action)
    return env


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import numpy as np

def make_parallel_env(config):

    def get_env_fn(config):
        def init_env(config):
            env = make_env(config.env_id, config, discrete_action=config.discrete_action)
            # env.seed(seed + rank * 1000)
            # np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if config.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], config)
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(config.n_rollout_threads)])