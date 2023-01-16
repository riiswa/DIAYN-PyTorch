import numpy as np
from .model import Discriminator
import torch
from torch.nn import functional as F
from torch import from_numpy, nn
from .replay_memory import Memory, Transition
import random
from copy import deepcopy
from torch.optim.adam import Adam
import itertools


class EvolutionaryPolicyNetwork(torch.nn.Module):
    def __init__(
            self,
            n_states,
            n_actions,
            action_bounds,
            n_hidden_filters=256,
            n_hidden_layers=2,
            patience=5,
            min_delta=1e-8,
            max_epochs=500
    ):
        super(EvolutionaryPolicyNetwork, self).__init__()
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
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs

    def forward(self, state):
        x = F.relu(self.input(state))
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        x = torch.tanh(self.output(x))
        return (x * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1])

    def mutate(self, states, actions):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        best_loss = float('inf')
        early_stop_counter = 0
        best_weights = None

        for epoch in range(500):
            outputs = self(states)
            loss = self.loss_fn(outputs, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < best_loss - self.min_delta:
                best_loss = loss
                early_stop_counter = 0
                best_weights = self.state_dict()
            else:
                early_stop_counter += 1
            if early_stop_counter >= self.patience:
                #print(f'Early stopping at epoch {epoch + 1} with loss {best_loss:.8f}')
                break
        #print(best_loss.item())
        return best_weights


class EvolutionaryAgent:
    def __init__(self, action_space, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_space = action_space

        torch.manual_seed(self.config["seed"])

        def create_for_n_skills(object):
            return [object() for _ in range(self.n_skills)]

        self.memories = create_for_n_skills(
            lambda: Memory(self.config["mem_size"] // self.n_skills, self.config["seed"]))
        self.policy_networks = create_for_n_skills(lambda: EvolutionaryPolicyNetwork(n_states=self.n_states,
                                                                                     n_actions=self.config["n_actions"],
                                                                                     action_bounds=self.config[
                                                                                         "action_bounds"],
                                                                                     # n_hidden_filters=self.config["n_hiddens"]
                                                                                     ).to(self.device)
                                                   )
        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.policy_opts = [
            Adam(policy_network.parameters(), lr=self.config["lr"]) for policy_network in self.policy_networks
        ]
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        self.p_z = torch.tensor(1 / self.n_skills + 1e-6)

    def choose_action(self, z, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action = self.policy_networks[z](states)
        return action.detach().cpu().numpy()[0]

    def intrinsic_reward(self, z, next_state):
        next_state = from_numpy(next_state).float().to(self.device)
        return torch.log(self.discriminator(next_state)[z].detach() + 1e-6) - torch.log(self.p_z)

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor(np.array([action])).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memories[z].add(state, z, done, action, next_state)

    def unpack(self, batch, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(batch_size, self.n_states).to(self.device)
        zs = torch.cat(batch.z).view(batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        next_states = torch.cat(batch.next_state).view(batch_size, self.n_states).to(self.device)

        return states, zs, dones, actions, next_states

    def train_discriminator(self):
        if sum([len(m) for m in self.memories]) < self.batch_size:
            return None
        memory = list(itertools.chain.from_iterable([memory.buffer for memory in self.memories]))

        sum_discriminator_loss = 0

        for _ in range(self.n_skills * 10):
            batch = random.sample(memory, self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)

            logits = self.discriminator(states)
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()
            sum_discriminator_loss += discriminator_loss.item()

        return -sum_discriminator_loss / (self.n_skills * 10)

    def train_skills(self, rewards):
        def f(x):
            return x ** 2 / 4 - x / 2 + 1 / 4

        ranking = np.argsort(rewards)
        for z in range(self.n_skills):
            rank = np.where(ranking == z)[0][0]
            #mutation_probability = (rank + 1) / self.n_skills * 0.5
            memory = deepcopy(self.memories[z])
            indexes = []
            #print("RANK", rank)
            for i, transition in enumerate(memory.buffer):
                output = self.discriminator(transition.next_state).detach()
                #print(output[z] - output.mean(), f(output[z] - output.mean()))
                e = f(output[z] - output.mean())
                if e > 0.25:
                    transition.action = from_numpy(np.array([self.action_space.sample()]))
                    indexes.append(i)
            for index in sorted(indexes, reverse=True):
                del self.memories[z].buffer[index]

            states, zs, dones, actions, next_states = self.unpack(memory.buffer, len(memory))
            self.policy_networks[z].mutate(states, actions)
