import random
import torch
import _pickle as pickle

class Arglist:
    def __init__(self):
        # self.USE_CUDA = torch.cuda.is_available()
        self.USE_CUDA = False
        if self.USE_CUDA:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.env_id = "simple_tag"
        self.model_name = "./test_model_2prey"
        self.predators_comm = False
        self.predators_comm_size = 8    # each agent sends a 1-hot-vector in this size to all teammates.
        # self.seed = 1
        # self.seed = random.randint(0, 1e5)
        self.n_rollout_threads = 1
        self.n_training_threads = 6
        self.buffer_length = int(1e6)
        self.num_runs = 30
        self.n_episodes = 25000
        self.episode_length = 25
        self.steps_per_update = 100
        self.batch_size = 1024
        self.n_exploration_eps = 25000
        self.init_noise_scale = 0.3
        self.final_noise_scale = 0.0
        self.save_interval = 1000
        self.hidden_dim = 64
        # self.hidden_dim = 32
        self.lr = 0.01
        self.tau = 0.01
        # self.agent_alg = "MADDPG"
        self.agent_alg = "DDPG"
        self.adversary_alg = "MADDPG" # choices=['MADDPG', 'DDPG']
        self.discrete_action = True
        self.load_model_path = "C:\\git\\torch_maddpg\\models\\simple_tag\\test_model_2prey\\run6\\model.pt"
        # self.load_model_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/test_model_2prey/run6/model.pt"

    def save(self, run_dir):
        file = open(run_dir / "arglist.txt", 'w')
        for arg in self.__dict__:
            file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
        file.close()

    def print_args(self):
        for arg in self.__dict__:
            print(arg + " = " + str(self.__dict__[arg]))