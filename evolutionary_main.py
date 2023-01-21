import gymnasium as gym
import torch
from matplotlib import pyplot as plt

from Brain import EvolutionaryAgent
from Common import Play, get_params
import numpy as np
from tqdm import tqdm
import multiprocess as mp


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    mp.set_start_method('spawn')
    params = get_params()

    test_env = gym.make("Ant-v4", exclude_current_positions_from_observation=False)
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make("Ant-v4", exclude_current_positions_from_observation=False)

    n = 500
    input_queue = mp.Queue(maxsize=n)
    output_queue = mp.Queue(maxsize=n)
    processes = []
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = EvolutionaryAgent(
        p_z=p_z,
        std_dev=0.05,
        input_queue=input_queue,
        output_queue=output_queue,
        processes=processes,
        n=n,
        **params
    )

    for _ in range(mp.cpu_count()):
        process = mp.Process(target=agent.worker, args=(input_queue, output_queue))
        process.start()
        processes.append(process)

    if params["do_train"]:

        min_episode = 0
        last_logq_zs = 0
        np.random.seed(params["seed"])
        env.observation_space.seed(params["seed"])
        env.action_space.seed(params["seed"])
        print("Training from scratch.")

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            agent.train_policy(env, z)
            state, info  = env.reset(seed=params["seed"])
            if isinstance(state, tuple):
                state = state[0]
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = 500
            for step in range(1, 1 + max_n_steps):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                final_state = next_state
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, terminated or truncated, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if terminated or truncated:
                    break
            agent.archive.append(torch.from_numpy(final_state[:2]))
            agent.z_archive.append(z)
            print(sum(logq_zses) / len(logq_zses))

            #
            # if episode % 10 == 0:
            #     trajectories = []
            #     for z in range(params["n_skills"]):
            #         state, info  = env.reset(seed=params["seed"])
            #         state = concat_state_latent(state, z, params["n_skills"])
            #         max_n_steps = 100
            #         trajectory = []
            #         for step in range(1, 1 + max_n_steps):
            #             action = agent.choose_action(state)
            #             next_state, reward, done = env.step(action)[:3]
            #             trajectory.append(next_state[:2])
            #             next_state = concat_state_latent(next_state, z, params["n_skills"])
            #             state = next_state
            #             if done:
            #                 break
            #         trajectories.append(trajectory)
            #
            #     plt.close()

        for _ in range(n):
            input_queue.put(None)

        for process in processes:
            process.join()

    else:
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
