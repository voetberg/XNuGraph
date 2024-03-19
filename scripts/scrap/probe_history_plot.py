import matplotlib.pyplot as plt
import json
import numpy as np

probe_history_hipmip = json.load(open("hipmip_trainer_history.json", "r"))
probe_history_tracks = json.load(open("track_trainer_history.json", "r"))

plt.close("all")
keys = probe_history_tracks.keys()
fig, subplots = plt.subplots(
    nrows=2, ncols=len(keys), sharey="col", sharex=False, figsize=(5 * len(keys), 10)
)

ylabels = ["Track identification", "Hip/Mip difference"]
for subplot, key in enumerate(keys):
    for col, history in enumerate([probe_history_tracks, probe_history_hipmip]):
        train = history[key][:-1]
        index = [i for i in range(len(train))]
        # handle the message passing ones

        if isinstance(train[0], list):
            train = np.array(train).T

            for message_index, message_step in enumerate(train):
                index = [i for i in range(len(message_step))]
                subplots[col, subplot].plot(
                    index, message_step, label=f"Message Step {message_index+1}"
                )

            subplots[col, subplot].legend()
        else:
            subplots[col, subplot].plot(index, train, label="Train", color="blue")

        subplots[col, subplot].set_title(key)
        subplots[col, 0].set_ylabel(ylabels[col])

fig.supxlabel("Training Epoch")
fig.supylabel("Loss")
fig.tight_layout()
plt.legend()
plt.savefig("dynamic_probe_loss.png")
