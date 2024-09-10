from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import torch
import pandas as pd

from AriTaxiRewardTransformer import AriTaxiRewardTransformer

my_taxi_env = gym.make("Taxi-v3")
my_taxi_env = AriTaxiRewardTransformer(my_taxi_env)

ppo_taxi_model = PPO.load("ppo_original_taxi_env_with_reward_wrapper", env=my_taxi_env)

out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0

o = out[0]

# while True:
for i in range(250):
    action, _states = ppo_taxi_model.predict(o, deterministic=False)
    o, r, d, _, _ = my_taxi_env.step(int(action))
    print("Took action "+str(int(action)))
    print("Received reward "+str(int(r)))
    ep_ret += r
    ep_len += 1
    # obs = out[0]
    my_taxi_env.render()
    if d:  # or (ep_len == max_ep_len):
        # logger.store(EpRet=ep_ret, EpLen=ep_len)
        # print("************* COMPLETED EPISODE ***************")
        print('Episode %d \t EpRet %.3f \t EpLen %d' % (i, ep_ret, ep_len))
        print("************* BEGINNING NEW EPISODE ***************")
        out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0
        o = out[0]
        print("Initial state is " + str(int(o)))
        # n += 1
