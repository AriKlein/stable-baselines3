from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import gymnasium as gym

# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = gym.make("Taxi-v3") # , render_mode='human') # , n_envs=4, seed=0)

# Frame-stacking with 4 frames
# vec_env = VecFrameStack(vec_env, n_stack=4)

ppo_taxi_model = PPO("MlpPolicy", vec_env, verbose=1)
ppo_taxi_model.learn(total_timesteps=int(500000), progress_bar=True)
ppo_taxi_model.save("ppo_taxi_with_pickup_six_actions")

my_taxi_env = vec_env.env.env.env

out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0

o = out[0]

# while True:
for i in range(1000):
    action, _states = ppo_taxi_model.predict(o, deterministic=False)
    # obs, rewards, dones, info \
    o, r, d, _, _ = my_taxi_env.step(int(action))
    ep_ret += r
    ep_len += 1
    # obs = out[0]
    my_taxi_env.render()
    if d:  # or (ep_len == max_ep_len):
        # logger.store(EpRet=ep_ret, EpLen=ep_len)
        print('Episode %d \t EpRet %.3f \t EpLen %d' % (i, ep_ret, ep_len))
        out, r, d, ep_ret, ep_len = my_taxi_env.reset(), 0, False, 0, 0
        o = out[0]
        print("Initial state is " + str(int(o)))
        # n += 1



# model = PPO.load("ppo_taxi", env=vec_env)
'''
obs, info = vec_env.reset()
# while True:
for i in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    # obs, rewards, dones, info \
    out = vec_env.step(int(action))
    obs = out[0]
    vec_env.env.render()
'''