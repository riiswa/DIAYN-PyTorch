import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from torch.utils.tensorboard import SummaryWriter


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


def check_bounds(coords, b=5):
    return np.all(np.logical_and(coords >= -b, coords <= b))


def coverage(trajectories, b=5):
    x, y = np.meshgrid(np.linspace(-b, b, 1000), np.linspace(-b, b, 1000))
    points = np.array((x.flatten(), y.flatten())).T

    mask = np.zeros_like(x, dtype=bool)

    for trajectory in trajectories:
        # Create a path object using the points
        path = Path(trajectory)
        # Check which points are contained within the trajectory
        mask_trajectory = path.contains_points(points)
        # Add the points contained within the trajectory to the mask array
        mask = np.logical_or(mask, mask_trajectory.reshape(x.shape))

    return np.count_nonzero(mask) / points.shape[0]


def plot_trajectories(trajectories, b=5):
    fig = plt.figure(figsize=(14, 14))
    for trajectory in trajectories:
        # Create a scatter plot of the points
        plt.plot(*zip(*trajectory), alpha=0.7)
        plt.scatter(*zip(*trajectory), alpha=0.7)

        # Remove the axes and grid
        # plt.axis('off')
        plt.grid(False)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

    return fig


def evaluate_agent(params, env, agent, writer: SummaryWriter, episode):
    trajectories = []
    for z in range(params["n_skills"]):
        state, info = env.reset(seed=params["seed"])
        state = concat_state_latent(state, z, params["n_skills"])
        max_n_steps = 500
        trajectory = []
        for step in range(1, 1 + max_n_steps):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            truncated = truncated or not check_bounds(next_state[:2])
            trajectory.append(next_state[:2])
            next_state = concat_state_latent(next_state, z, params["n_skills"])
            state = next_state
            if terminated or truncated:
                break
        trajectories.append(trajectory)

    writer.add_scalar("coverage", coverage(trajectories), global_step=episode)
    writer.add_figure("trajectories", plot_trajectories(trajectories), global_step=episode)

    plt.close()
