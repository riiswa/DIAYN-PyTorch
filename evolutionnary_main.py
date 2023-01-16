import gym
from Brain import EvolutionaryAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
from torch import isnan

import multiprocess as mp

if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"])

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = EvolutionaryAgent(env.action_space, **params)
    #logger = Logger(agent, **params)

    min_episode = 0
    last_logq_zs = 0
    np.random.seed(params["seed"])
    #env.seed(params["seed"])
    env.observation_space.seed(params["seed"])
    env.action_space.seed(params["seed"])
    print("Training from scratch.")

    def sample(z):
        state = env.reset(seed=params["seed"])
        if isinstance(state, tuple):
            state = state[0]
        sample_reward = 0

        max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
        for step in range(1, 1 + max_n_steps):
            action = agent.choose_action(z, state)
            next_state, reward, done = env.step(action)[:3]
            agent.store(state, z, done, action, next_state)
            reward = agent.intrinsic_reward(z, next_state)
            if not isnan(reward):
                sample_reward += reward
            state = next_state
            if done:
                break

        return sample_reward, agent.memories[z]

    for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
        with mp.Pool(mp.cpu_count()) as p:
            rewards, memories = list(zip(*p.map(sample, range(params["n_skills"]))))
            agent.memories = memories
            print("log q", agent.train_discriminator())
            agent.train_skills(rewards)



