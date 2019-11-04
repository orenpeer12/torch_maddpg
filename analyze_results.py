import matplotlib.pyplot as plt
import json

path_to_summary = "C:\\git\\torch_maddpg\\models\\simple_tag\\test_model\\run3\\logs\\summary.json"
with open(path_to_summary) as json_file:
    data = json.load(json_file)
data.keys()
for loss in data.keys():
    plt.figure(loss)
    plt.plot([ep_loss[2] for ep_loss in data[loss]])
    plt.show()
