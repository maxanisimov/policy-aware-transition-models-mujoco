import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.transform_observation import GrayscaleObservation, ResizeObservation
import numpy as np
import logging
import time

class GrayScaleEnv(gym.Env):
    """
    Custom environment wrapper to convert observations to grayscale.
    This is a simple wrapper that uses the GrayscaleObservation from gymnasium.
    """
    def __init__(self, env):
        super(GrayScaleEnv, self).__init__()
        self.env = env
        # self.observation_space = GrayscaleObservation(env.observation_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation_space(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation_space(obs), reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the environment
env = GrayscaleObservation(gym.make('CarRacing-v3'), keep_dim=False) # Convert to grayscale
eval_env = GrayscaleObservation(gym.make('CarRacing-v3', render_mode='human')) # For evaluation with rendering

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# Add action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

policy_kwargs = dict(
    features_extractor_class=None,  # use default NatureCNN
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=[256, 256],           # two hidden layers of 256 units
    activation_fn="ReLU"
)

# Create the DDPG agent
model = DDPG(
    "MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    batch_size=128,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    action_noise=action_noise,
    # policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./ddpg_cr_tensorboard/",
    device='cpu'
)

# Create an evaluation callback
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path='./ddpg_carracing_best_model/',
    log_path='./ddpg_carracing_eval_logs/',
    eval_freq=50_000,  # Evaluate every 50,000 steps
    n_eval_episodes=10,
    deterministic=True,
    render=True,
)

# Train the agent
logger.info("Starting training...")
model.learn(
    total_timesteps=2_000_000,
    # callback=eval_callback
)

# Save the model
logger.info("Saving the model...")
model.save("ddpg_carracing_0306")

### Evaluation
logger.info("Starting evaluation...")
num_eval_episodes = 10
max_eval_steps = 1_000
sleep_time = 0.01

eval_total_rewards = []
for ep in range(num_eval_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    for step in range(max_eval_steps):
        # 1) Query model for action. Use deterministic=True for evaluation.
        action, _ = model.predict(obs, deterministic=True)

        # 2) Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 3) Render the current frame
        #    - If env was created with render_mode="human", this opens a window automatically
        #    - If render_mode="rgb_array", this returns an array you can save/view
        if hasattr(env, "render"):
            frame = env.render()  # returns None (human) or np.ndarray (rgb_array)

            # If rgb_array, you might want to show or save frames manually:
            if isinstance(frame, np.ndarray):
                # Example: show frame via matplotlib (slow!)
                # import matplotlib.pyplot as plt
                # plt.imshow(frame)
                # plt.axis("off")
                # plt.pause(0.0001)

                # Or write it to disk as an image:
                # from PIL import Image
                # Image.fromarray(frame).save(f"frame_ep{ep}_step{step}.png")

                pass

        # 4) If the episode ended, break
        if terminated or truncated:
            break

        # 5) (Optional) Slow down rendering so you can actually see it
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info(f"Episode {ep+1} finished ─ total_reward = {total_reward:.2f}")
    eval_total_rewards.append(total_reward)

logger.info(f"Average reward over {num_eval_episodes} episodes: {np.mean(eval_total_rewards):.2f}")

# Optionally, close the environment
env.close()
eval_env.close()
