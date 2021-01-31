import os

import gym
import glob
from core.DDQN_Agent import DDQN_Agent
from core.DQN_Agent import DQN_Agent
from core.FQTDQN_Agent import FQTDQN_Agent
from core.utils.utils import *
from datetime import datetime as dt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


def save_logs(path, mode, timestamp, list):
    Path(path + mode).mkdir(parents=True, exist_ok=True)
    f = open(path + mode + '/log_' + timestamp + '.txt', 'w')
    for l in list:
        f.writelines(l + '\n')
    f.close()


def train(mode, parameters_path, logs_path, npy_path=None, plt_path=None, load=None):
    # Chosen environment
    env = gym.make('MountainCar-v0')
    in_features = len(env.observation_space.sample())
    num_actions = env.action_space.n

    # FOR PLOTTING
    plt_avg_last_100_rewards = []
    plt_epsilon = []

    # HYPERPARAMETERS
    batch_size = 128
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.99

    target_update = 20
    memory_size = 300000
    learning_rate = 0.001
    num_episodes = 5000

    # UTILS
    save_weight_th = -110
    log_every = 500

    if mode == 'DQN':
        agent = DQN_Agent(num_actions=num_actions, in_features=in_features,
                          epsilon=eps_start, eps_min=eps_end, eps_decay=eps_decay,
                          gamma=gamma, learning_rate=learning_rate,
                          batch_size=batch_size, memory_size=memory_size)
    elif mode == 'FQTDQN':
        agent = FQTDQN_Agent(num_actions=num_actions, in_features=in_features,
                             epsilon=eps_start, eps_min=eps_end, eps_decay=eps_decay,
                             gamma=gamma, learning_rate=learning_rate,
                             batch_size=batch_size, memory_size=memory_size)

        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.target_net.eval()
    elif mode == 'DDQN':
        agent = DDQN_Agent(num_actions=num_actions, in_features=in_features,
                           epsilon=eps_start, eps_min=eps_end, eps_decay=eps_decay,
                           gamma=gamma, learning_rate=learning_rate,
                           batch_size=batch_size, memory_size=memory_size)

        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.target_net.eval()


    # START TRAINING
    all_timesteps_loss = []
    total_rewards = []
    goal_achieved = 0
    log_list = []
    game_solved = 0
    tot_timesteps = 0
    timestamp = dt.now().strftime("%d%m%Y-%H%M%S")

    for episode in range(1, num_episodes + 1):
        render = False

        if episode % log_every == 0:
            print("----------------------------\nEpisode {} started..".format(episode))
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        if episode == 1:
            eps = agent.strategy.get_exploration_rate()
            plt_epsilon.append(eps)
        else:
            eps = agent.strategy.update_exploration_rate()
            plt_epsilon.append(eps)

        while not done:
            step += 1
            tot_timesteps += 1

            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action.item())

            if done and step < 200:
                goal_achieved += 1
                reward = 0

            if render:
                env.render()

            episode_reward += reward

            agent.push_in_memory(
                Experience(torch.tensor([state]), action, torch.tensor([reward]), torch.tensor([next_state]),
                           torch.tensor([done])))

            state = next_state

            loss = agent.learn()
            all_timesteps_loss.append(loss)

            if step % target_update == 0 and (isinstance(agent, FQTDQN_Agent) or isinstance(agent, DDQN_Agent)):
                agent.hard_update_target_net()

        total_rewards.append(episode_reward)
        if len(total_rewards) < 100:
            avg_last_100_rewards = -np.inf
            plt_avg_last_100_rewards.append(avg_last_100_rewards)
        else:
            avg_last_100_rewards = np.mean(total_rewards[-100:])
            plt_avg_last_100_rewards.append(avg_last_100_rewards)

        if (episode % log_every) == 0:
            print(
                'Episode: {} ->\t Reward: {} - AvgLast100Reward: {} - Duration: {} timesteps.\n\t\t Epsilon: {} - Loss: {}\n\t\tGoals {}/{}\n----------------------------\n'.format(
                    episode, np.round(episode_reward), avg_last_100_rewards, step,
                    eps, loss, goal_achieved, log_every))

            log_list.append(
                'Episode: {} ->\t Reward: {} - AvgLast100Reward: {} - Duration: {} timesteps.\n\t\t Epsilon: {} - Loss: {}\n\t\tGoals {}/{}\n----------------------------\n'.format(
                    episode, np.round(episode_reward), avg_last_100_rewards, step,
                    eps, loss, goal_achieved, log_every))
            goal_achieved = 0

        if avg_last_100_rewards >= save_weight_th:
            save_weight_th = avg_last_100_rewards
            print('<<<<<<<<<<<<<<<<<<<<<<<<< PROBLEM SOLVED! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            game_solved += 1

            if isinstance(agent, FQTDQN_Agent) or isinstance(agent, DDQN_Agent):
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'loss': loss,
                },
                    parameters_path + mode + '/{}_[ep={}]_[avglast100r={}]_[decay={}].h5'.format(timestamp,
                                                                                                 episode,
                                                                                                 avg_last_100_rewards,
                                                                                                 eps_decay))
                print('Weights saved.')
            else:
                torch.save({
                    'episode': episode,
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'loss': loss,
                },
                    parameters_path + mode + '/{}_[ep={}]_[avglast100r={}]_[decay={}].h5'.format(timestamp,
                                                                                                 episode,
                                                                                                 avg_last_100_rewards,
                                                                                                 eps_decay))
                print('Weights saved.')

    env.close()

    print('G A M E    S O L V E D  [{}]  T I M E S.'.format(game_solved))

    save_logs(logs_path, mode, timestamp, log_list)

    plt_all_episodes_rewards = total_rewards
    plt_all_timesteps_loss = all_timesteps_loss

    plt_win_th = -110 * np.ones(num_episodes)

    if npy_path is not None:
        save_np(plt_avg_last_100_rewards, npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'avg100')
        save_np(plt_all_episodes_rewards, npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'ep_rewards')
        save_np(plt_win_th, npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'win_th')
        save_np(plt_all_timesteps_loss, npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'loss')
        save_np(np.arange(num_episodes), npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'episodes')
        save_np(np.arange(tot_timesteps), npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'timesteps')
        save_np(plt_epsilon, npy_path + mode, timestamp + '_[decay=' + str(eps_decay) + ']', 'eps')

    plot(x1=np.arange(num_episodes), y1=[plt_all_episodes_rewards, plt_avg_last_100_rewards, plt_win_th], \
         l1=['Episodes Reward', 'Avg Last 100 Rewards', 'Winning Condition'], x1_label='Episode', y1_label='Reward', \
         x2=np.arange(tot_timesteps), y2=[plt_all_timesteps_loss], \
         l2=['Loss'], x2_label='Timestep', y2_label='Loss', \
         title2='Loss Plot', path=plt_path + mode, name=timestamp + '_[decay=' + str(eps_decay) + ']_performances',
         save=True)

    plot(x1=np.arange(num_episodes), y1=[plt_epsilon], \
         l1=['Epsilon'], x1_label='Episodes', \
         title1='Epsilon Decay', \
         path=plt_path + mode, name=timestamp + '_[decay=' + str(eps_decay) + ']_eps', save=True)

    return
