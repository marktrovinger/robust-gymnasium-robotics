import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "Ant-v4"
args.noise_factor = "action"
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO



def main():
    env = gym.make(args.env_name, render_mode="human")
    #env_vec = make_vec_env(args.env_name)

    model = PPO("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=25000)
    
if __name__ == "__main__":
    main()