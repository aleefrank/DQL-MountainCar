import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, in_features, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=48)
        self.out = nn.Linear(in_features=48, out_features=num_actions)  # move left/right/stay

    def forward(self, t):
        t = t.float()
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        y = self.out(t)

        return y

