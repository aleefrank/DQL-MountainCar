import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

from collections import namedtuple
import torch


def plot(x1=None, x2=None, y1=[], y2=[], l1=[], l2=[], x1_label='', x2_label='', y1_label='', y2_label='', colors=[],
         title1='', title2='',
         figsize=[7, 4], path=None, name=None, save=False):
    for i in range(len(y1)):
        if not isinstance(y1[i], np.ndarray):
            y1[i] = np.array(y1[i])

    plt.figure(figsize=figsize)
    if save:
        to_pickle = []

    if (x1 is not None) and (x2 is not None):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        if colors:
            for i, j, k in zip(y1, l1, colors): axes[0].plot(x1, i, label=j, color=k)
            axes[0].set_xlabel(x1_label)
            axes[0].set_ylabel(y1_label)
            axes[0].set_title(title1)
            axes[0].legend(loc="upper right")
            to_pickle.append(axes[0])
            for i, j, k in zip(y2, l2), colors: axes[1].plot(x2, i, label=j, color=k)
            axes[1].set_xlabel(x2_label)
            axes[0].set_ylabel(y2_label)
            axes[1].set_title(title2)
            axes[1].legend(loc="upper right")
            to_pickle.append(axes[1])
        else:
            for i, j in zip(y1, l1): axes[0].plot(x1, i, label=j)
            axes[0].set_xlabel(x1_label)
            axes[0].set_title(title1)
            axes[0].legend(loc="upper right")
            to_pickle.append(axes[0])
            for i, j in zip(y2, l2): axes[1].plot(x2, i, label=j)
            axes[1].set_xlabel(x2_label)
            axes[1].set_title(title2)
            axes[1].legend(loc="upper right")
            to_pickle.append(axes[1])

        fig.tight_layout()
    else:
        if colors:
            for i, j, k in zip(y1, l1, colors):
                linestyle = '-'
                if j.startswith('Win') or j.startswith('Defeat'):
                    linestyle = '--'

                plt.plot(x1, i, label=j, color=k, linestyle=linestyle)
        else:
            for i, j in zip(y1, l1):

                linestyle = '-'
                if j.startswith('Win') or j.startswith('Defeat'):
                    linestyle = '--'

                plt.plot(x1, i, label=j, linestyle=linestyle)

        plt.xlabel(x1_label)
        plt.ylabel(y1_label)
        plt.title(title1)
        plt.legend(loc="upper right")
        to_pickle.append(plt)

    if save:
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.savefig(path + '/' + name + '.png')
            # with open(path + '/' + name + '.pkl'), 'wb') as f:  # should be 'wb' rather than 'w'
            #    for item in to_pickle: pickle.dump(item, f)

    plt.show()


def save_np(x, path, timestamp, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(path + '/' + timestamp + '_' + name + '.npy', x)


# Credits for this representation with namedtuple: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state', 'done')
)


# Converts batch of Experiences to Experience of batches
# Credits : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def transpose(experiences):
    batch = Experience(*zip(*experiences))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
