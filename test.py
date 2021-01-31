import gym
import glob
from core.model import DQN
from core.utils.utils import *


def test(mode, name, parameters_path, plt_results=False):
    # Chosen environment
    env = gym.make('MountainCar-v0')
    in_features = len(env.observation_space.sample())
    num_actions = env.action_space.n

    try:
        files = glob.glob(parameters_path + '/' + mode + '/*')
        # latest file contains the best parameters
        best_param = max(files)
    except ValueError:
        print("Couldn't find parameters file in path {}".format(parameters_path + '/' + mode + '/'))
        raise
    try:
        f = torch.load(best_param)
        test_net = DQN(in_features=in_features, num_actions=num_actions)
        test_net.load_state_dict(f['policy_net_state_dict'])
        test_net.eval()
    except OSError:
        print("File not found.")
        raise

    try:
        num_episodes = 100
        n_goal = 0
        scores = []
        game_lost = -200

        print('\nGAME STARTED!\n')
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            for step in range(200):
                #env.render()
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor([state])
                action = test_net(state).argmax(dim=1)

                next_state, reward, done, _ = env.step(action.item())
                episode_reward += reward
                state = next_state
                if done:
                    scores.append(episode_reward)
                    if episode_reward > -200:
                        n_goal += 1
                        print('Episode {}: Goal achieved in {} steps'.format(episode, episode_reward))
                    else:
                        print(episode_reward)
                        print('Episode {}: Goal not achieved.'.format(episode))
                    break
        env.close()
    except:
        print('\nSomething occurred during Testing..')

    if plt_results:
        game_lost = game_lost * np.ones(num_episodes)
        mean_score = np.mean(scores) * np.ones(num_episodes)
        num_episodes = np.arange(num_episodes)

        plot(x1=num_episodes,
             y1=[scores, mean_score, game_lost], \
             l1=['Score', 'Avarage Score', 'Defeat Threshold'], x1_label='Episode', y1_label='Score', \
             path='./figures', name=mode + name + '_Agent_test',
             save=True, colors=['tab:blue', 'tab:orange', 'tab:red'])

    return
