import random
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, num_states, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_states, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=48)
        self.out = nn.Linear(in_features=48, out_features=num_actions)  # move left/right/stay

    def forward(self, t):
        t = t.float()
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        y = self.out(t)

        return y


# Credits for this representation of ReplayMemory: https://deeplizard.com/learn/video/PyQNfsGUnQA
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


