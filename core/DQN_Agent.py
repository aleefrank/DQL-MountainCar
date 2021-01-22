import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.model import ReplayMemory, DQN
from core.utils.utils import extract_tensors

class DQN_Agent():
    def __init__(self, num_actions, num_states, \
                 epsilon, eps_min, eps_decay, \
                 gamma, learning_rate, \
                 batch_size, memory_size):

        self.name = 'DQN'
        self.num_actions = num_actions
        self.num_states = num_states

        self.strategy = EpsGreedyStrategy(epsilon, eps_min, eps_decay)

        self.gamma = gamma
        self.lr = learning_rate

        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_size)

        self.policy_net = DQN(num_states=num_states, num_actions=num_actions)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

    def get_agent_name(self):
        return self.name

    # explore or exploit based on eps
    def get_action(self, state, epsilon):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state])
        if epsilon > np.random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action])

        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1)

    # methods to facilitate memory access
    def memory_can_provide_sample(self, batch_size):
        return self.replay_memory.can_provide_sample(batch_size)

    def push_in_memory(self, experience):
        return self.replay_memory.push(experience)

    def memory_sample(self, batch_size):
        return self.replay_memory.sample(batch_size)

    def perform_exp_replay(self):
        if self.memory_can_provide_sample(self.batch_size):
            batches = self.memory_sample(self.batch_size)
            states, actions, rewards, next_states, dones = extract_tensors(batches)
            mask_not_ending_states = (dones.type(torch.bool) == False)

            curr_state_q_val = self.get_curr_state_q_val(states, actions)
            next_state_q_val = self.get_next_state_q_val(next_states)
            target_q_val = rewards + (self.gamma * next_state_q_val * mask_not_ending_states)

            loss = self.optimize_policy_net(curr_state_q_val, target_q_val)

            return loss.detach().item()
        return 0

    def get_curr_state_q_val(self, states, actions):
        return self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    def get_next_state_q_val(self, next_states):
        return self.policy_net(next_states).max(dim=1)[0].detach()

    def optimize_policy_net(self, curr, exp):
        loss = F.mse_loss(curr, exp)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # used in case of training interruption or test
    def load_parameters(self, file):
        f = torch.load(file)
        self.policy_net.load_state_dict(f['policy_net_state_dict'])
        self.get_target_net().load_state_dict(f['target_net_state_dict'])
        self.optimizer.load_state_dict(f['optimizer_state_dict'])
        self.policy_net.train()
        self.target_net.eval()

        # just for printing them out if interrupted
        episode = f['episode']
        loss = f['loss']

        return episode, loss

class EpsGreedyStrategy():
    def __init__(self, eps, eps_min, eps_decay):
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def update_exploration_rate(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return self.eps

    def get_exploration_rate(self):
        return self.eps
