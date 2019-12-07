import _pickle as pickle
import json
from json import JSONEncoder

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
        self.env_id = "simple_tag"
        self.use_prey_controller = True
        self.prey_max_speed = 0.5
        self.pred_max_speed = 1
        self.pred_acc = 3
        self.prey_acc = 2

        self.num_landmarks = 0
        self.num_predators = 1
        self.num_prey = 2
        self.shaping = True

        ## General agrs:
        shape = "sum" if self.shaping else "no"
        self.model_name = \
            "./" + str(self.num_prey) + "prey_" + str(self.num_predators) + "pred_noCom_" + shape + "Shape_noLand"
        # self.model_name = "./play1"
        self.comments = "controller prey."

        ## Algorithm args:
        self.predators_comm = False
        self.predators_comm_size = 4 if self.predators_comm else 0  # each agent sends a 1-hot-vector in this size to all teammates.
        self.symbolic_comm = False
        self.n_rollout_threads = 1
        self.n_training_threads = 6
        self.buffer_length = int(1e6)
        self.num_runs = 15
        self.n_episodes = 30000
        # self.n_episodes = 25000
        # self.n_episodes = 50000
        self.episode_length = 25
        self.n_time_steps = self.n_episodes * self.episode_length
        self.steps_per_update = 100
        self.steps_per_eval = 10000
        self.num_steps_in_eval = 10000
        self.batch_size = 1024
        self.n_exploration_steps = self.n_time_steps
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
        self.run_dir = ""

    # def save(self, run_dir):
    #     self.load_model_path = run_dir / "model.pt"
    #     text_file = open(run_dir / "arglist.txt", 'w')
    #
    #     with open(run_dir / "arglist.pkl", 'wb') as output:
    #         pickle.dump(self, output)
    #     for arg in self.__dict__:
    #         text_file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
    #     text_file.close()


    def save_args(self, run_dir):
        self.run_dir = run_dir
        text_file = open(run_dir / "arglist.txt", 'w')
        with open(run_dir / "arglist.json", 'w') as output:
            json.dump(self.__dict__, output)

        for arg in self.__dict__:
            text_file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
        text_file.close()

    def load_args(self, path_to_args):
        with open(path_to_args / "arglist.json") as f:
            json_data = json.load(f)
        self.__dict__ = {}
        for k in json_data:
            self.__setattr__(k, json_data[k])

    # def load_args(self, path_to_args):
    #     import pathlib
    #     pathlib.PureWindowsPath(path_to_args)
    #     with open(pathlib.PureWindowsPath(path_to_args), 'rb') as input:
    #         loaded = input.read()
    #     return pickle.loads(loaded)

    def print_args(self):
        for arg in self.__dict__:
            print(arg + " = " + str(self.__dict__[arg]))

