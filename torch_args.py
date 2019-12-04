import random
import torch
import _pickle as pickle
import copy

class Arglist:
    def __init__(self):
        ## System agrs:
        # self.USE_CUDA = torch.cuda.is_available()

        self.USE_CUDA = False
        if self.USE_CUDA:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        ## Environment args:
        self.num_landmarks = 0
        self.num_prey = 2
        self.num_predators = 1
        self.env_id = "simple_tag"
        self.shaping = True
        self.use_prey_controller = True
        self.prey_max_speed = 0.5
        self.pred_max_speed = 1
        self.pred_acc = 3
        self.prey_acc = 2

        ## General agrs:
        self.model_name = "./2prey_1pred_noCom_sumShape_noLand_LONG"
        # self.model_name = "./play1"
        self.comments = "controller prey.."

        ## Algorithm args:
        self.predators_comm = False
        self.predators_comm_size = 4 if self.predators_comm else 0  # each agent sends a 1-hot-vector in this size to all teammates.
        self.symbolic_comm = False
        self.n_rollout_threads = 1
        self.n_training_threads = 6
        self.buffer_length = int(1e6)
        self.num_runs = 20
        # self.n_episodes = 25000
        self.n_episodes = 50000
        self.episode_length = 25
        self.steps_per_update = 100
        self.batch_size = 1024
        self.n_exploration_eps = 25000
        self.init_noise_scale = 0.3
        self.final_noise_scale = 0.0
        self.save_interval = 1000
        self.hidden_dim = 64
        self.gamma = 0.95
        self.lr = 0.01
        self.tau = 0.01
        # self.agent_alg = "DDPG"
        self.agent_alg = "CONTROLLER"
        self.adversary_alg = "MADDPG" # choices=['MADDPG', 'DDPG']
        self.discrete_action = False
        # self.load_model_path = "C:\\git\\torch_maddpg\\models\\simple_tag\\results_predators\\test_model_max_not_min\\run0\\model.pt"
        # self.load_model_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/1prey_thin_obs_full_a/run2/model.pt"
        self.load_model_path = "/home/oren/PycharmProjects/torch_maddpg/models/simple_tag/1prey_thinObs_critic_comm/run0/model.pt"

    def save(self, run_dir):
        self.load_model_path = run_dir / "model.pt"
        text_file = open(run_dir / "arglist.txt", 'w')

        with open(run_dir / "arglist.pkl", 'wb') as output:
            pickle.dump(self, output)
        for arg in self.__dict__:
            text_file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
        text_file.close()

    def load_args(self, path_to_args):
        from pathlib import Path, PosixPath
        with open(path_to_args._str, 'rb') as input:
            loaded = pickle.load(input)
        return pickle.loads(loaded)

    def print_args(self):
        for arg in self.__dict__:
            print(arg + " = " + str(self.__dict__[arg]))

