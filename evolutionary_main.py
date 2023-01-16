import gym
from Brain import SACAgent, EvolutionaryAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
#import mujoco_py
import multiprocess as mp


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


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

    n = 50
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

    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            raise NotImplemented()
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            #env.seed(params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            agent.train_policy(env, z)
            state = env.reset(seed=params["seed"])
            if isinstance(state, tuple):
                state = state[0]
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)[:3]
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break
            print(sum(logq_zses) / len(logq_zses))

            """
            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                       env.np_random. __getstate__(),
                       env.observation_space.np_random.__get_state__(),
                       env.action_space.np_random.__get_state__(),
                       *agent.get_rng_states(),
                       )
            """

        for _ in range(n):
            input_queue.put(None)

        for process in processes:
            process.join()

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
