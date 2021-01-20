import torch

from core.DQN_Agent import DQN_Agent
from core.model import DQN


class DDQN_Agent(DQN_Agent):
    def __init__(self, num_actions, num_states, \
                 epsilon, eps_min, eps_decay, \
                 gamma, learning_rate, \
                 batch_size, memory_size):
        DQN_Agent.__init__(self, num_actions, num_states, \
                           epsilon, eps_min, eps_decay, \
                           gamma, learning_rate, \
                           batch_size, memory_size)
        self.name = 'DDQN'
        self.target_net = DQN(num_states=num_states, num_actions=num_actions)

    def get_next_state_q_val(self, next_states):
        return self.target_net(next_states).gather(1, torch.max(self.policy_net(next_states), 1)[1].unsqueeze(1)).squeeze(1)

    def hard_update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
