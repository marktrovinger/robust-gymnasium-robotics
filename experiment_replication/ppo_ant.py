import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config
#import gymnasium as gym
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from action_injection_wrapper import VectorActionInjection, ActionInjectionWrapper



def main():
    robust_args = get_config().parse_args()
    robust_args.env_name = "Ant-v4"
    robust_args.noise_factor = "state"
    #env_name = "Ant-v4"
    env = gym.make(robust_args.env_name, render_mode="rgb_array")
    #env = gym.make(env_name, render_mode="rgb_array")
    check_env(env)
    #env = ActionInjectionWrapper(env, robust_input={"robust_type" : "state","robust_config": robust_args})
    #env_vec = make_vec_env(args.env_name)

    model = PPO("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=25000)
    
if __name__ == "__main__":
    main()