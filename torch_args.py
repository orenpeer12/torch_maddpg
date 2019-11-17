import random
import torch
import _pickle as pickle
import copy


class Arglist:
    def __init__(self):
        # self.USE_CUDA = torch.cuda.is_available()
        self.comments = ""
        self.USE_CUDA = False
        if self.USE_CUDA:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.num_landmarks = 2
        self.num_prey = 1
        self.num_predators = 3
        self.env_id = "simple_tag"
        # self.env_id = "simple_speaker_listener"
        self.model_name = "./1prey_thin_obs_full_a"
        self.predators_comm = False
        self.predators_comm_size = 4    # each agent sends a 1-hot-vector in this size to all teammates.
        self.n_rollout_threads = 1
        self.n_training_threads = 6
        self.buffer_length = int(1e6)
        self.num_runs = 30
        self.n_episodes = 25000
        self.episode_length = 25
        self.steps_per_update = 100
        self.batch_size = 1024
        self.n_exploration_eps = 25000  # this specify the
        self.init_noise_scale = 0.3
        self.final_noise_scale = 0.0
        self.save_interval = 1000
        self.hidden_dim = 64
        self.gamma = 0.95
        self.lr = 0.01
        self.tau = 0.01
        # self.agent_alg = "MADDPG"
        self.agent_alg = "DDPG"
        self.adversary_alg = "MADDPG" # choices=['MADDPG', 'DDPG']
        self.discrete_action = True
        # self.load_model_path = "C:\\git\\torch_maddpg\\models\\simple_tag\\results_predators\\test_model_max_not_min\\run0\\model.pt"
        self.load_model_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/1prey_thin_obs_full_a/run2/model.pt"

    def save(self, run_dir):
        text_file = open(run_dir / "arglist.txt", 'w')

        with open(run_dir / "arglist.pkl", 'wb') as output:
            pickle.dump(self, output)
        for arg in self.__dict__:
            text_file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
        text_file.close()

    def load(self, path_to_args):

        with open(path_to_args, 'rb') as input:
            loaded = input.read()
        return pickle.loads(loaded)

    def print_args(self):
        for arg in self.__dict__:
            print(arg + " = " + str(self.__dict__[arg]))

