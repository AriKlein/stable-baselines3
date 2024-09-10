# Aaron (Ari) Klein
# Principal Engineer, AI/ML Wireless Systems, Kenyi Technologies

"""
Custom RewardWrapper to modify the reward structure of the taxi environment so that I can optimize with PPO algo:
  - Reward +10 for correct pickup so that agent can more quickly learn to do correct pickups without needing to
    randomly do a full correct pickup -> dropoff sequence before seeing any reward.
  - Penalize dropoff after pickup (reward = -12) to avoid infinite reward loop with pickup->dropoff->pickup->dropoff...
  - Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent doesn't learn to largely avoid
    doing pickups and dropoffs
"""

import gymnasium as gym
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


"""
Custom RewardWrapper to modify the reward structure of the taxi environment so that I can optimize with PPO algo:
  - Reward +10 for correct pickup so that agent can more quickly learn to do correct pickups without needing to 
    randomly do a full correct pickup -> dropoff sequence before seeing any reward.
  - Penalize dropoff after pickup (reward = -12) to avoid infinite reward loop with pickup->dropoff->pickup->dropoff...
  - Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent doesn't learn to largely avoid
    doing pickups and dropoffs
"""
class AriTaxiRewardTransformer(gym.RewardWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the Reward wrapper."""
        gym.RewardWrapper.__init__(self, env)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        previous_state = self.env.unwrapped.s
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(previous_state, action), terminated, truncated, info

    def reward(self, previous_state, action):
        """Transforms the reward using callable :attr:`f`.

        Args:
            previous_state: The state before taking the action,
            obs: The state AFTER taking the previous action,
            next_state: The state after taking the action

        Returns:
            The transformed reward
        """

        reward = -1
        [taxi_row, taxi_col, pass_idx, dest_idx] = self.env.unwrapped.decode(previous_state)
        taxi_loc = (taxi_row, taxi_col)
        locs = self.env.unwrapped.locs
        if action == 4:  # pickup
            # Reward +10 for correct pickup so that agent can more quickly learn to do correct pickups without needing
            #    to randomly do a full correct pickup -> dropoff sequence before seeing any reward.
            if pass_idx < 4 and taxi_loc == locs[pass_idx]:  # did correct pickup.  reward = +10
                reward = 10
            else:  # reduce penalty for pickup at wrong location
                # Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent
                # doesn't learn to largely avoid doing pickups and dropoffs
                reward = -2
        elif action == 5:  # dropoff
            if (taxi_loc == locs[dest_idx]) and pass_idx == 4:  # did correct dropoff.  reward as before
                reward = 20
            elif (taxi_loc in locs) and pass_idx == 4:  # Penalty of -12 to avoid pickup->dropoff->pickup->dropoff loop
                # Penalize dropoff after pickup (reward = -12) to avoid infinite reward loop with pickup->dropoff->pickup->dropoff...
                reward = -12
            else:  # reduce penalty for dropoff at wrong location
                # Reduce penalties for illegal pickups and dropoffs from -10 to -2 so that the agent
                # doesn't learn to largely avoid doing pickups and dropoffs
                reward = -2
        return reward
