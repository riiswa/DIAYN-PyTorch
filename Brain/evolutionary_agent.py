from collections import deque

import numpy as np
import torch
from torch import from_numpy
from torch.optim.adam import Adam
import scipy.stats as ss

from Brain.model import PolicyNetwork, Discriminator
from Brain.replay_memory import Memory, Transition
from main import concat_state_latent
from torch.utils.tensorboard import SummaryWriter


def normalized_rank(rewards):
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm


def average_distance(point, points_list, k):
    distances = torch.norm(point.unsqueeze(0) - points_list, p=2, dim=1)
    _, indices = distances.sort()
    k_nearest_indices = indices[:k]
    return distances[k_nearest_indices].mean()


class MyLogger:
    def __init__(self):
        self.writer = SummaryWriter()

    def add_loss(self, loss):
        self.writer.add_scalar("logq(z|s)", loss)



class EvolutionaryAgent:
    def __init__(self,
                 p_z,
                 std_dev,
                 input_queue,
                 output_queue,
                 processes,
                 n,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.std_dev = std_dev
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processes = processes
        self.n = n

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.cross_ent_loss = torch.nn.CrossEntropyLoss()
        self.policy_opt = Adam(self.policy_network.parameters(), lr=0.005)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        self.archive = deque([], maxlen=self.n_skills * 2)
        self.z_archive = deque([], maxlen=self.n_skills * 2)

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor(np.array([action])).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)

            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            return -discriminator_loss.item()

    def policy_weights(self):
        return [param.data for param in self.policy_network.parameters()]

    def perturb_policy_weights(self, weights):
        for param, weight in zip(self.policy_network.parameters(), weights):
            param.data += weight.float() * self.std_dev

    def policy_noise(self):
        return np.array(
            [torch.normal(0, 1, size=param.data.size()) for param in self.policy_network.parameters()],
            dtype=object
        )

    def intrinsic_reward(self, z, next_state):
        next_state = from_numpy(next_state).float().to(self.device)
        return torch.nan_to_num(torch.log(self.discriminator(next_state)[z].detach() + 1e-6)) - \
               torch.log(torch.tensor(1 / self.n_skills))

    def evaluate(self, env, z):
        state, info = env.reset(seed=self.config["seed"])
        if isinstance(state, tuple):
            state = state[0]
        state = concat_state_latent(state, z, self.n_skills)
        episode_reward = 0
        max_n_steps = 500
        for step in range(1, 1 + max_n_steps):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)[:3]
            with torch.no_grad():
                episode_reward += self.intrinsic_reward(z, next_state)
            if terminated or truncated:
                break
            final_state = next_state
            state = concat_state_latent(next_state, z, self.n_skills)

        return episode_reward, torch.from_numpy(final_state[:2])

    def evaluate_noisy(self, env, z, noise):
        old_dict = self.policy_network.state_dict()
        self.perturb_policy_weights(noise)
        reward, final_state = self.evaluate(env, z)
        self.policy_network.load_state_dict(old_dict)
        return reward, final_state

    def worker(self, input_queue, output_queue):
        while True:
            params = input_queue.get()
            if params is not None:
                env, z = params
                seed = torch.randint(high=int(1e6), size=(1,))
                torch.manual_seed(seed.item())
                noise = self.policy_noise()

                pos_reward, pos_final_sate = self.evaluate_noisy(env, z, noise)
                neg_reward, neg_final_state = self.evaluate_noisy(env, z, -noise)

                output_queue.put(([pos_reward, neg_reward], (pos_final_sate, neg_final_state), seed))
            else:
                break

    def train_policy(self, env, z, ):
        noises = []
        rewards = []
        for _ in range(self.n):
            self.input_queue.put((env, z))

        for _ in range(self.n):
            process_rewards, process_final_state, process_seed = self.output_queue.get()
            if any(skill != z for skill in self.z_archive):
                final_states = torch.stack([state for state, skill in zip(self.archive, self.z_archive) if skill != z])
                for i in range(2):
                    process_rewards[i] *= average_distance(process_final_state[i], final_states, self.n_skills)
            torch.manual_seed(process_seed)
            noise = self.policy_noise()
            noises.extend([noise, -noise])
            rewards.extend(process_rewards)

        rewards = normalized_rank(rewards)

        self.policy_opt.zero_grad()

        for idx, weight in enumerate(self.policy_network.parameters()):
            upd_weights = torch.zeros(weight.data.shape)
            for noise, reward in zip(noises, rewards):
                upd_weights += reward * noise[idx]
            upd_weights = upd_weights / (self.n * self.std_dev)
            weight.grad = (-upd_weights).float()

        self.policy_opt.step()
