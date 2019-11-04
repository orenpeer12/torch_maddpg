class Arglist:
    def __init__(self):
        self.env_id = "simple_tag"
        self.model_name = "./test_folder/test_model"
        self.seed = 1
        self.n_rollout_threads = 1
        self.n_training_threads = 1
        self.buffer_length = int(1e6)
        self.n_episodes = 10000 # 25000
        self.episode_length = 25
        self.steps_per_update = 100
        self.batch_size = 1024
        self.n_exploration_eps = 10000 # 25000
        self.init_noise_scale = 0.3
        self.final_noise_scale = 0.0
        self.save_interval = 1000
        self.hidden_dim = 64
        self.lr = 0.01
        self.tau = 0.01
        self.agent_alg = "MADDPG"
        self.adversary_alg = "MADDPG" # choices=['MADDPG', 'DDPG']
        self.discrete_action = True
