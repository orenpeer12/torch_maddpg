import _pickle as pickle
import json
import time
import torch

class Arglist:
    def __init__(self):
        ######################
        #### System agrs: ####
        ######################
        self.USE_CUDA = torch.cuda.is_available()
        # self.USE_CUDA = False
        # self.USE_CUDA = True
        if self.USE_CUDA:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.n_rollout_threads = 1
        self.n_training_threads = 6
        ###########################
        #### Environment args: ####
        ###########################
        self.env_id = "simple_tag"
        self.bound_world = True
        self.use_prey_controller = False
        # self.prey_max_speed = 0.5
        # self.prey_max_speed = 1.3
        # self.prey_max_speed = 0.6
        # self.rand_prey_speed = True
        # self.prey_max_speed = 1
        self.prey_max_speed = 'random'
        self.pred_max_speed = 1
        self.pred_acc = 3
        # self.prey_acc = 2
        # self.prey_acc = 4
        self.prey_acc = 3
        self.num_landmarks = 0
        ##
        self.num_prey = 2
        self.num_predators = 2
        self.shaping = False
        # IL
        self.use_IL = False  # imitation learning flag.
        self.predators_comm = False
        self.predators_comm_size = 1 if self.predators_comm else 0  # each agent sends a 1-hot-vector in this size to all teammates.
        self.IL_inject_every = 5000 if self.use_IL else -1
        self.IL_decay = 0.6
        self.IL_amount = 500
        #######################
        #### General agrs: ####
        #######################
        comm = "_withCom{}".format(self.predators_comm_size) if self.predators_comm else "_noCom"
        speed = "_randomPreySpeed" if self.prey_max_speed == 'random' else "_SameSpeedPrey" if self.prey_max_speed == self.pred_max_speed else "_SlowPrey" if self.prey_max_speed < self.pred_max_speed else "_FastPrey"
        controller = "_controllerPrey" if self.use_prey_controller else "_DDPGprey"
        shape = "_sumShape" if self.shaping else "_noShape"
        IL_str = "_withIL" if self.use_IL else "_noIL"
        extra_str = ""
        walls = "_withWalls" if self.bound_world else "_noWalls"
        entities_str = str(self.num_prey) + "prey_" + str(self.num_predators) + "pred_" + \
                       str(self.num_landmarks) + "landmarks" + walls

        # self.model_name = "./" + entities_str + comm + shape + IL_str + controller + speed + extra_str
        self.model_name = "./debug"
        self.comments = ""
        #########################
        #### Algorithm args: ####
        #########################
        self.symbolic_comm = False
        # Run parameters
        self.buffer_length = int(1e4)
        self.num_runs = 8
        # self.n_episodes = 40000
        self.n_episodes = 20000
        # self.n_episodes = 50000
        self.episode_length = 40
        self.n_time_steps = self.n_episodes * self.episode_length
        self.steps_per_update = 100
        self.steps_per_eval = 10000
        self.num_steps_in_eval = 5000
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
        self.agent_alg = "CONTROLLER" if self.use_prey_controller else "DDPG"
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
        self.run_dir = run_dir if type(run_dir) is str else str(run_dir)
        text_file = open(run_dir / "arglist.txt", 'w')
        with open(run_dir / "arglist.json", 'w') as output:
            json.dump(self.__dict__, output)

        for arg in self.__dict__:
            text_file.write(arg + " = " + str(self.__dict__[arg]) + "\n")
        text_file.write(time.asctime())
        text_file.close()

    def load_args(self, path_to_args):
        with open(path_to_args / "arglist.json") as f:
            json_data = json.load(f)
        self.__dict__ = {}
        for k in json_data:
            self.__setattr__(k, json_data[k])

    def print_args(self):
        for arg in self.__dict__:
            print(arg + " = " + str(self.__dict__[arg]))

