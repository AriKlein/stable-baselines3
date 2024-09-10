"""""
Author: Aaron (Ari) Klein, Principal Engineer, AI/ML Wireless Systems, Kenyi Technologies

Uses Stable Baselines 3 PPO algorithm to train MLP policy for the gym taxi problem

Uses my custom reward structure so that I can optimize with PPO algo:
  - Reward +10 for correct pickup so that agent can more quickly learn to do correct pickups without needing to
    randomly do a full correct pickup -> dropoff sequence before seeing any reward.
  - Penalize dropoff after pickup (reward = -12) to avoid infinite reward loop with pickup->dropoff->pickup->dropoff...
  - Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent doesn't learn to largely avoid
    doing pickups and dropoffs

"""

# import my custom reward structure for Taxi env
from AriTaxiRewardTransformer import AriTaxiRewardTransformer

from stable_baselines3 import PPO
import gymnasium as gym

# Here we are also multi-worker training (n_envs=4 => 4 environments)
my_taxi_env = gym.make("Taxi-v3") # , render_mode='human') # , n_envs=4, seed=0)
my_taxi_env = AriTaxiRewardTransformer(my_taxi_env)

# Frame-stacking with 4 frames

ppo_taxi_model = PPO("MlpPolicy", my_taxi_env, verbose=1)
ppo_taxi_model.learn(total_timesteps=int(500000), progress_bar=True)
ppo_taxi_model.save("ppo_original_taxi_env_with_reward_wrapper")

# my_taxi_env = vec_env.env.env.env

out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0

o = out[0]

# while True:
episode_num = 1
step_num = 1
max_episodes = 10
max_steps = 250
while episode_num <= max_episodes and step_num <= max_steps:
    action, _states = ppo_taxi_model.predict(o, deterministic=False)
    o, r, d, _, _ = my_taxi_env.step(int(action))
    step_num = step_num + 1
    ep_ret += r
    ep_len += 1
    my_taxi_env.render()
    if d:  # or (ep_len == max_ep_len):
        # logger.store(EpRet=ep_ret, EpLen=ep_len)
        print('Episode %d \t EpRet %.3f \t EpLen %d' % (episode_num, ep_ret, ep_len))
        if episode_num < max_episodes:
            print("************* BEGINNING NEW EPISODE ***************")
            out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0
            o = out[0]
            print("Initial state is " + str(int(o)))
        episode_num = episode_num + 1
