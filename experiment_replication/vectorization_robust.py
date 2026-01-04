import robust_gymnasium as gym

envs = gym.make_vec("Ant-v4", num_envs=3, vectorization_mode="sync", wrappers=(gym.wrappers.TimeAwareObservation,))
envs = gym.wrappers.vector.ClipReward(envs, min_reward=0.2, max_reward=0.8)

print(envs.num_envs)