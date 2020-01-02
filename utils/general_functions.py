from pathlib import Path
import os
import gc
import time
import numpy as np


class running_env_manager():
    def __init__(self, MODE):
        self.DEBUG = True if MODE == 'DEBUG' else False
        self.run_dir = ""

    def prep_running_env(self, config,run_num):
        if not self.DEBUG:
            gc.collect()
            model_dir = Path('./models') / config.env_id / config.model_name
            if not model_dir.exists():
                curr_run = 'run0'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                                 model_dir.iterdir() if
                                 str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run0'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)

            self.run_dir = model_dir / curr_run
            log_dir = self.run_dir / 'logs'
            if not log_dir.exists():
                os.makedirs(log_dir)

            config.save_args(self.run_dir)
            if run_num is 0: config.print_args()
            # logger = SummaryWriter(str(log_dir))
        else:
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("results will not be saved!")
            print("results will not be saved!")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            if run_num is 0: config.print_args()
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("results will not be saved!")
            print("results will not be saved!")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")
            print("DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! DEBUG MODE!! ")


    def printProgressBar(self, iteration, start_time, total, prefix='', suffix='', length=100, fill='█', done=False):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """

        percent = "{0:.1f}".format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end='')
        # Print New Line on Complete #
        if iteration == total:
            ending = " || Total runtime: {0:.1f} minutes.".format((time.time() - start_time) / 60) + "\n"
            print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix + ending), end='')
            print("\n")

