from core.utils.utils import *
from test import test

from train import train


def main():
    parameters_path = './parameters/'
    logs_path = './logs/'
    plt_path = './plt/'
    npy_path = './npy/'

    load_npy = False
    plt_compare = False

    print("\n\n* * *    W E L C O M E   T O   M O U N T A I N   C A R   D Q N    * * *\n\n")
    while True:
        choice = input(
            "Do you want to Train or Test the Network?\n[1] - Train\n[2] - Test\n[3] - Plot Results\n\n[4] - Terminate\n")

        if choice not in ['1', '2', '3', '4']:
            print("Option [{}] not allowed.\nAllowed options: [1] - [2]\nTry again...".format(choice))
            continue
        elif choice == '1':
            print("- You selected Train -")
            mode_choice = input(
                "Select the Training Mode:\n[1] - DQN\n[2] - DQN with Fixed Q-Targets\n[3] - Double DQN\n")
            if mode_choice not in ['1', '2', '3']:
                print(
                    "Option [{}] not allowed.\nAllowed options: [1] - [2] - [3]\nTerminating the program...".format(
                        mode_choice))
                continue
            if mode_choice == '1':
                mode = "DQN"
            elif mode_choice == '2':
                mode = "FQTDQN"
            elif mode_choice == '3':
                mode = "DDQN"
            print("- Training Mode: {}".format(mode))
            input("Press any Key to start Training..")
            train(mode=mode, parameters_path=parameters_path, logs_path=logs_path, plt_path=plt_path,
                  npy_path=npy_path, load=False)
        elif choice == '2':
            print("- You selected Test -")
            mode_choice = input(
                "\tSelect the Test Mode:\n\t[1] - DQN\n\t[2] - DQN with Fixed Q-Targets\n\t[3] - Double DQN\n")
            if mode_choice not in ['1', '2', '3']:
                print(
                    "Option [{}] not allowed.\nAllowed options: [1] - [2] - [3]\nTry again...".format(
                        mode_choice))
                continue
            if mode_choice == '1':
                mode = "DQN"
            elif mode_choice == '2':
                mode = "FQTDQN"
            elif mode_choice == '3':
                mode = "DDQN"
            print("- Test Mode: {}".format(mode))
            name = input("Save .png as: ")
            input("Press any Key to start Testing..")
            test(mode=mode, name=name, parameters_path=parameters_path, plt_results=True)
        elif choice == '3':
            mode_choice = input(
                '\tSelect the Agent s result you want to plot:\n\t[1] - DQN\n\t[2] - DQN with Fixed Q-Targets\n\t[3] - Double DQN\n')
            if mode_choice not in ['1', '2', '3']:
                print(
                    "Option [{}] not allowed.\nAllowed options: [1] - [2] - [3]\nTry again...".format(
                        mode_choice))
                continue
            if mode_choice == '1':
                mode = "DQN"
            elif mode_choice == '2':
                mode = "FQTDQN"
            elif mode_choice == '3':
                mode = "DDQN"
            load_npy = True
            filename = input('Insert the identifier of the file you want to plot the results (format: DDMMYYYY-hhmmss_[decay=*]):')

        elif choice == '4':
            #plt_compare = True
            print('\nTerminating....\n')
            break

        if load_npy:
            starts_with = mode + '/' + filename
            try:
                num_episodes = np.load(npy_path + starts_with + '_episodes.npy')
                plt_all_episodes_rewards = np.load(npy_path + starts_with + '_ep_rewards.npy')
                plt_avg_last_100_rewards = np.load(npy_path + starts_with + '_avg100.npy')
                plt_win_th = np.load(npy_path + starts_with + '_win_th.npy')
            except FileNotFoundError:
                print("file not found.")
            plot(x1=num_episodes,
                 y1=[plt_all_episodes_rewards.tolist(), plt_avg_last_100_rewards.tolist(), plt_win_th.tolist()], \
                 l1=['Episodes Reward', 'Avarage Score', 'Winning Condition'], x1_label='Episode', y1_label='Score', \
                 path='./figures', name=mode + '_Agent',
                 save=True)

        if plt_compare:
            #Path(parameters_path + mode).mkdir(parents=True, exist_ok=True)
            #if load:
            #    list_of_files = glob.glob(parameters_path + mode + '/*')
            #    latest_file = max(list_of_files, key=os.parameters_path.getctime)
            #    try:
            #    except OSError:
            #        print("file not found")
            #        raise
            dqn_starts_with = 'DQN/19012021-110922_[decay=0.99]_'
            fqtdqn_starts_with = 'FQTDQN/18012021-190116_[decay=0.99]_'
            ddqn_starts_with = 'DDQN/19012021-190116_[decay=0.99]_'
            num_episodes = np.load(npy_path + dqn_starts_with + 'episodes.npy')
            plt_win_th = np.load(npy_path + dqn_starts_with + 'win_th.npy')
            plt_avg100_dqn = np.load(npy_path + dqn_starts_with + 'avg100.npy')
            plt_avg100_fqtdqn = np.load(npy_path + fqtdqn_starts_with + 'avg100.npy')
            plt_avg100_ddqn = np.load(npy_path + ddqn_starts_with + 'avg100.npy')

            plot(x1=num_episodes, y1=[plt_avg100_dqn, plt_avg100_fqtdqn, plt_avg100_ddqn, plt_win_th], \
                 l1=['DQN Agent', 'FQTDQN Agent', 'DDQN Agent', 'Winning Condition'], x1_label='Episode',
                 y1_label='Avarage Score', \
                 path='./figures', name='Results',
                 save=True, colors=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'])


if __name__ == '__main__':
    main()
