import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
import time
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate DDPG policy.")
parser.add_argument(
    "--model_path", "-mp", required=True, help="Path to policy model"
)
parser.add_argument(
    "--env_name", "-en", required=True, help="Environment name (e.g., HalfCheetah-v5)"
)
parser.add_argument(
    "--num_episodes", "-ne", type=int, default=10, help="Number of episodes to evaluate"
)
args = parser.parse_args()

model_path = args.model_path
env_name = args.env_name
model = DDPG.load(model_path, device='cpu')
env = gym.make(env_name, render_mode="human")

num_episodes = args.num_episodes
sleep_time = 0.01

if __name__ == "__main__":

    total_rewards = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done: # in HalfCheetah-v5, default duration of episode is 1000 steps 
            # 1) Query model for action. Use deterministic=True for evaluation.
            action, _ = model.predict(obs, deterministic=True)

            # 2) Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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

            # # 4) (Optional) Slow down rendering so you can actually see it
            # if sleep_time > 0:
            #     time.sleep(sleep_time)

        total_rewards.append(total_reward)
        logger.info(f"Episode {ep+1} finished ─ total_reward = {total_reward:.2f}")

    logger.info(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()
