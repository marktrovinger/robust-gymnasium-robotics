import robust_gymnasium as gym
from robust_gymnasium.core import ActType, Env
from robust_gymnasium import Space
from robust_gymnasium.vector import VectorWrapper, VectorEnv
from robust_gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate
from robust_gymnasium.wrappers import transform_action
from robust_gymnasium.core import ActType, ObsType, WrapperActType
from robust_gymnasium.spaces import Box, Space

class VectorActionInjection(VectorWrapper):
    def __init__(
            self,
            env: VectorEnv,
            robust_input: dict,
            action_space: Space | None = None,
        ):
        super().__init__(env)
        self.env = env
        self.robust_input = robust_input

    def step(self, actions):
        self.robust_input["action"] = actions
        observation, reward, terminated, truncated, info = self.env.step(self.robust_input)
        return observation, reward, terminated, truncated, info

class ActionInjection(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            robust_input: dict,
            #action_space: Space | None = None,
        ):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.robust_input = robust_input

    def step(self, action):
        self.robust_input["action"] = action
        #observation, reward, terminated, truncated, info = self.env.step(self.robust_input)
        #return observation, reward, terminated, truncated, info
        return super().step(self.robust_input)