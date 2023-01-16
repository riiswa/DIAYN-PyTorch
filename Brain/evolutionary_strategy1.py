import numpy as np
import torch
from torch import from_numpy, nn
from torch.optim.adam import Adam
import scipy.stats as ss
from torch.nn import functional as F

from Brain.model import Discriminator
from Brain.replay_memory import Memory, Transition


def normalized_rank(rewards):
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm


class PolicyNetwork(torch.nn.Module):
    def __init__(
            self,
            n_states,
            n_actions,
            action_bounds,
            n_hidden_filters=32,
            n_hidden_layers=2,
    ):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_hidden_layers = n_hidden_layers
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.input = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.hidden = [
            nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
            for _ in range(self.n_hidden_layers)
        ]
        self.output = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.input(state))
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        x = torch.tanh(self.output(x))
        return (x * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1])


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
        self.policy_networks = [PolicyNetwork(n_states=self.n_states,
                                              n_actions=self.config["n_actions"],
                                              action_bounds=self.config["action_bounds"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device) for _ in
                                range(self.config["n_skills"])]

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.cross_ent_loss = torch.nn.CrossEntropyLoss()
        self.policy_opts = [Adam(policy_network.parameters(), lr=0.01) for policy_network in self.policy_networks]
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states, z):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action = self.policy_networks[z](states)
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

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)

            logits = self.discriminator(states)
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            return -discriminator_loss.item()

    def policy_weights(self, z):
        return [param.data for param in self.policy_networks[z].parameters()]

    def perturb_policy_weights(self, weights, z):
        for param, weight in zip(self.policy_networks[z].parameters(), weights):
            param.data += weight.float() * self.std_dev

    def policy_noise(self, z):
        return np.array(
            [torch.normal(0, 1, size=param.data.size()) for param in self.policy_networks[z].parameters()],
            dtype=object
        )

    def intrinsic_reward(self, z, next_state):
        next_state = from_numpy(next_state).float().to(self.device)
        return torch.nan_to_num(torch.log(self.discriminator(next_state)[z].detach() + 1e-6)) - \
               torch.log(torch.tensor(1 / self.n_skills))

    def evaluate(self, env, z):
        state = env.reset(seed=self.config["seed"])
        if isinstance(state, tuple):
            state = state[0]
        # state = concat_state_latent(state, z, self.n_skills)
        episode_reward = 0
        max_n_steps = 300
        for step in range(1, 1 + max_n_steps):
            action = self.choose_action(state, z)
            next_state, reward, done = env.step(action)[:3]
            with torch.no_grad():
                episode_reward += self.intrinsic_reward(z, next_state)
            # state = concat_state_latent(next_state, z, self.n_skills)

        return episode_reward

    def evaluate_noisy(self, env, z, noise):
        old_dict = self.policy_networks[z].state_dict()
        self.perturb_policy_weights(noise, z)
        reward = self.evaluate(env, z)
        self.policy_networks[z].load_state_dict(old_dict)
        return reward

    def worker(self, input_queue, output_queue):
        while True:
            params = input_queue.get()
            if params is not None:
                env, z = params
                seed = torch.randint(high=int(1e6), size=(1,))
                torch.manual_seed(seed.item())
                noise = self.policy_noise(z)

                pos_reward = self.evaluate_noisy(env, z, noise)
                neg_reward = self.evaluate_noisy(env, z, -noise)

                output_queue.put(((pos_reward, neg_reward), seed))
            else:
                break

    def train_policy(self, env, z):
        noises = []
        rewards = []
        for _ in range(self.n):
            self.input_queue.put((env, z))

        for _ in range(self.n):
            process_rewards, process_seed = self.output_queue.get()
            torch.manual_seed(process_seed)
            noise = self.policy_noise(z)
            noises.extend([noise, -noise])
            rewards.extend(process_rewards)

        rewards = normalized_rank(rewards)

        self.policy_opts[z].zero_grad()

        for idx, weight in enumerate(self.policy_networks[z].parameters()):
            upd_weights = torch.zeros(weight.data.shape)
            for noise, reward in zip(noises, rewards):
                upd_weights += reward * noise[idx]
            upd_weights = upd_weights / (self.n * 0.05)
            weight.grad = (-upd_weights).float()

        self.policy_opts[z].step()
