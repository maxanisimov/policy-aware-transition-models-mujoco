from stable_baselines3 import DDPG
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

env = gym.make('BipedalWalker-v3')
n_actions = env.action_space.shape[-1]

model = DDPG(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=200_000,
    learning_starts=10_000,
    tau=0.005,
    gamma=0.98,
    action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    gradient_steps=1,
    train_freq=1,
    verbose=1,
    policy_kwargs=dict(net_arch=[400, 300]),
)

model.learn(
    total_timesteps=1e6,
    log_interval=100,
    progress_bar=True
)

### Evaluation
eval_env = gym.make('BipedalWalker-v3', render_mode='human')
num_eval_episodes = 10
total_rewards = []
for ep in range(num_eval_episodes):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    total_rewards.append(total_reward)
    print(f"Episode {ep + 1} finished ─ total_reward = {total_reward:.2f}")

eval_env.close()

print('Average evaluation reward over 10 episodes:', np.mean(total_rewards))
