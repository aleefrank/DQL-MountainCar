from core.DQN_Agent import DQN_Agent
from core.model import DQN


class FQTDQN_Agent(DQN_Agent):
    def __init__(self, num_actions, in_features, \
                 epsilon, eps_min, eps_decay, \
                 gamma, learning_rate, \
                 batch_size, memory_size):
        DQN_Agent.__init__(self, num_actions, in_features, \
                          epsilon, eps_min, eps_decay, \
                          gamma, learning_rate, \
                          batch_size, memory_size)
        self.name = 'FQTDQN'
        self.target_net = DQN(in_features=in_features, num_actions=num_actions)

    def get_next_state_q_val(self, next_states):
        return self.target_net(next_states).max(dim=1)[0].detach()

    def hard_update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
