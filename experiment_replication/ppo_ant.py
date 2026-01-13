import robust_gymnasium as gym
from robust_gymnasium.configs.robust_setting import get_config
args = get_config().parse_args()
args.env_name = "Ant-v4"
args.noise_factor = "action"
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from action_injection_wrapper import VectorActionInjection, ActionInjection



def main():
    robust_args = get_config().parse_args()
    robust_args.env_name = "Ant-v4"
    robust_args.noise_factor = "state"
    env = gym.make(args.env_name, render_mode="rgb_array")
    # TODO: SB3 complains that the type is wrong here; should be a Gymnasium Env, but is a
    # <class 'action_injection_wrapper.ActionInjectionWrapper'>, need to figure out what is
    # going on here
    assert isinstance(env, gym.Env), "Environment must be a gym.Env"
    env = gym.wrappers.ClipReward(env, -1.0, 1.0)
    env = ActionInjection(env, robust_input={"robust_type" : "state","robust_config": robust_args})
    assert isinstance(env, gym.Env), "Environment must be a gym.Env"
    #env_vec = make_vec_env(args.env_name)

    model = PPO("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=25000)
    
if __name__ == "__main__":
    main()