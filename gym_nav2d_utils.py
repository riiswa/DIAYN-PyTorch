import gym
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mplPath


def gym_nav2d_coverage(trajectories):
    x, y = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    points = np.array((x.flatten(), y.flatten())).T

    mask = np.zeros_like(x, dtype=bool)

    for trajectory in trajectories:
        # Create a path object using the points
        path = mplPath.Path(trajectory)
        # Check which points are contained within the trajectory
        mask_trajectory = path.contains_points(points)
        # Add the points contained within the trajectory to the mask array
        mask = np.logical_or(mask, mask_trajectory.reshape(x.shape))

    return np.count_nonzero(mask) / points.shape[0]


def gym_nav2d_plot(trajectories, file_name):
    for trajectory in trajectories:
        # Create a scatter plot of the points
        plt.plot(*zip(*trajectory), alpha=0.8)
        plt.scatter(*zip(*trajectory), alpha=0.8)

        # Remove the axes and grid
        plt.axis('off')
        plt.grid(False)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title(gym_nav2d_coverage(trajectories))
        plt.savefig(file_name)


if __name__ == "__main__":
    trajectories = []
    for e in range(10):
        env = gym.make('gym_nav2d:nav2dVeryEasy-v0')
        obs = env.reset()
        cumulated_reward = 0
        i = 0
        done = False
        trajectory = []
        while not done and i <= 100:
            i += 1
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)  # take a random action
            trajectory.append(obs[:2])
            cumulated_reward += rew
        trajectories.append(trajectory)

        env.close()


