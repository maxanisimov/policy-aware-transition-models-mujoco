import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the environment
env = gym.make('HalfCheetah-v5')

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# Add action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Create the DDPG agent
model = DDPG(
    policy="MlpPolicy",
    policy_kwargs=dict(
        net_arch=[400, 300, 200], # hidden layers and their sizes # 400x300x200 gives the avg eval reward over 10 episodes of 156.44
        # activation_fn="ReLU"
    ),
    env=env,
    learning_rate=1e-4,
    buffer_size=2e-5,
    learning_starts=10_000,
    tau=0.005,
    gamma=0.99,
    action_noise=action_noise,
    gradient_steps=1,
    verbose=1,
    batch_size=256,
    train_freq=(1, "episode"),
    device='cpu',
    seed=seed,
    tensorboard_log="./ddpg_halfcheetah_tensorboard/",
)

# Train the agent
logger.info("Starting training...")
model.learn(total_timesteps=1_000_000)

# Save the model
logger.info("Saving the model...")
model.save("ddpg_halfcheetah")

# Optionally, close the environment
env.close()

### Evaluation
logger.info("Starting evaluation...")
num_eval_episodes = 10
max_eval_steps = 1_000
sleep_time = 0.01

env = gym.make("HalfCheetah-v5", render_mode="human")
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
env.close()
