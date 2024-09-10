"""""
Requires xlsxwriter and pandas as well as Stable Baselines 3

Author: Aaron (Ari) Klein, Principal Engineer, AI/ML Wireless Systems, Kenyi Technologies

 - Evaluates the trained MLP policy on the gym taxi problem

 - Export greedy deterministic policy (e.g., the action with maximum probability for each state) to command line
    
 - Exports policy distribution for each state to an Excel spreadsheet
"""

"""
Custom RewardWrapper to modify the reward structure of the taxi environment so that I can optimize with PPO algo:
  - Reward +10 for correct pickup so that agent can more quickly learn to do correct pickups without needing to 
    randomly do a full correct pickup -> dropoff sequence before seeing any reward.
  - Penalize dropoff after pickup (reward = -12) to avoid infinite reward loop with pickup->dropoff->pickup->dropoff...
  - Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent doesn't learn to largely avoid
    doing pickups and dropoffs
"""
from AriTaxiRewardTransformer import AriTaxiRewardTransformer

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import pandas as pd

from stable_baselines3.common.monitor import Monitor

# Method to pretty-print the policies:
# - Prints a policy map for each passenger location given that the passenger is not yet in the taxi, so the taxi should be trying to get to and pick up the passenger
# - Prints a policy map for each destination location given that the passenger is already in the taxi, so the taxi should be trying to get to the destination and drop off the passenger
def pretty_print_policy(taxi, local_policy):

    MAP = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ]

    direction_repr = {1:' 🡑 ', 2:' 🡒 ', 3:' 🡐 ', 0:' 🡓 ', 4:' + ', 5:' - ', None:' ⬤ '}

    # Print policies for states where we are trying to get to passenger, so dest_idx is irrelevant, as long as not = pass_idx

    print('Passenger not in taxi, pass at Red (top left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 0, 1)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at Green (Top Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 1, 0)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at yellow (Bottom Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 2, 0)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger not in taxi, pass at Blue (Bottom Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 3, 0)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()


    # Print policies for states where we already have passenger and are trying to get to destination, so pass_idx is always 4

    print('Passenger in taxi, Dest = Red (Top Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 0)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()

    print('Passenger in taxi, Dest = Green (Top Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 1)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()


    print('Passenger in taxi, Dest = Yellow (Bottom Left):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 2)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()


    print('Passenger in taxi, Dest = Blue (Bottom Right):')
    for row in range(5):
        for col in range(5):
            state = taxi.encode(row, col, 4, 3)
            print(MAP[row+1][2*col],end='')
            print(direction_repr[local_policy[state]],end='')
        print()


def export_policy_to_excel(model, my_env):

    possible_actions = ["South", "North", "East", "West", "Pickup", "Dropoff"]

    excel_df = {}
    excel_df_keys_state = ['Taxi Row', 'Taxi Column', 'Passenger Index', 'Destination Index']
    excel_df_keys_actions = []
    for ii in possible_actions:
        excel_df_keys_actions.append(['pi('+ii+')'])
    # excel_df_keys_actions.append = excel_df_keys + ['V']

    greedy_policy = np.zeros(my_env.observation_space.n)

    for ii in range(len(excel_df_keys_state)):
        excel_df[excel_df_keys_state[ii]] = []

    for ii in range(len(excel_df_keys_actions)):
        excel_df[excel_df_keys_actions[ii][0]] = []

    excel_df['Sum Probs'] = []
    excel_df['V'] = []
    excel_df['Highest Probability Action'] = []


    #for obs in range(my_env.observation_space.n):
    for dest_idx in range(4):
        for pass_idx in range(5):
            for row_idx in range(5):
                for col_idx in range(5):
                    obs = my_env.unwrapped.encode(row_idx, col_idx, pass_idx, dest_idx)
                    # Decode state and put into Excel DF
                    obs_decoded = list(my_env.unwrapped.decode(obs))
                    for ii in range(len(obs_decoded)):
                        excel_df[excel_df_keys_state[ii]].append(obs_decoded[ii])

                    # Pass state to model
                    # obs = torch.as_tensor(obs, dtype=torch.float32)
                    # action, _states = model.predict(obs, deterministic=True)
                    # greedy_policy[obs] = action

                    # get action probabilities from observations according to policy network
                    p = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs[0]

                    # get value function from observations according to value function network
                    features = model.policy.extract_features(model.policy.obs_to_tensor(obs)[0])
                    latent_pi, latent_vf = model.policy.mlp_extractor(features)
                    v = model.policy.value_net(latent_vf)

                    for ii in range(len(p)):
                        excel_df[excel_df_keys_actions[ii][0]].append(float(p[ii]))

                    excel_df['Sum Probs'].append(float(sum(p)))
                    excel_df['V'].append(float(v))
                    greedy_action_index = int(np.argmax(p.cpu().detach().numpy()))
                    excel_df['Highest Probability Action'].append(possible_actions[greedy_action_index])

                    greedy_policy[obs] = greedy_action_index

    df = pd.DataFrame(excel_df)

    writer = pd.ExcelWriter('ppo_taxi_custom_rewards.xlsx', engine="xlsxwriter")
    df.to_excel(writer, sheet_name='ppo_taxi_custom_rewards')
    writer.close()

    pretty_print_policy(my_env, greedy_policy)


def main():
    max_episode_len = 100
    my_taxi_env = gym.make("Taxi-v3", render_mode='human')
    my_taxi_env = AriTaxiRewardTransformer(my_taxi_env)
    my_taxi_env = gym.wrappers.TimeLimit(my_taxi_env, max_episode_steps=max_episode_len)
    my_taxi_env = Monitor(my_taxi_env, allow_early_resets=True)

    ppo_taxi_model = PPO.load("ppo_original_taxi_env_with_reward_wrapper", env=my_taxi_env)

    export_policy_to_excel(ppo_taxi_model, my_taxi_env)

    out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0

    o = out[0]

    episode_num = 1
    step_num = 1
    max_episodes = 10
    max_steps = 250
    while episode_num <= max_episodes and step_num <= max_steps:
        action, _states = ppo_taxi_model.predict(o, deterministic=False)
        o, r, d, _, _ = my_taxi_env.step(int(action))
        step_num = step_num+1
        #print("Took action "+str(int(action)))
        #print("Received reward "+str(int(r)))
        ep_ret += r
        ep_len += 1
        my_taxi_env.render()
        if d:  # or (ep_len == max_ep_len):
            # print("************* COMPLETED EPISODE ***************")
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (episode_num, ep_ret, ep_len))
            if episode_num < max_episodes:
                print("************* BEGINNING NEW EPISODE ***************")
                out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0
                o = out[0]
                print("Initial state is " + str(int(o)))
            episode_num = episode_num + 1


main()
