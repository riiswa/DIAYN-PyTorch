import torch
from torch import nn
from torch.nn import functional as F
from time import time

import gym
import numpy as np

import optuna


class PolicyNetwork(torch.nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256, n_hidden_layers=3):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_hidden_layers = n_hidden_layers
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.input = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.hiddens = [nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters) for _ in range(self.n_hidden_layers)]
        self.output = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, state):
        x = F.relu(self.input(state))
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x = torch.tanh(self.output(x))
        return (x * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1])


def objective(trial):

    env = gym.make("BipedalWalker-v3")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    env.action_space.seed(123)
    env.observation_space.seed(123)

    np.random.seed(123)

    torch.manual_seed(123)

    n_hidden_filters = trial.suggest_int('n_hidden_filters', 2, 9)
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 10)


    skill = PolicyNetwork(n_states, n_actions, action_bounds, n_hidden_filters=2**n_hidden_filters, n_hidden_layers=n_hidden_layers)


    states = torch.cat([torch.from_numpy(np.array([env.observation_space.sample()])) for _ in range(256)])

    with torch.no_grad():
        targets = skill(states)

    for i in range(64):
        targets[i] = torch.from_numpy(env.action_space.sample())

    loss_fn = torch.nn.MSELoss()

    lr = trial.suggest_int("learning_rate", -6, -2)

    optimizer = torch.optim.AdamW(skill.parameters(), lr=10**lr)

    with torch.no_grad():
        outputs = skill(states)
        loss = loss_fn(outputs, targets)

        print(loss.item())


    patience = trial.suggest_int('patience', 2, 10)
    min_delta = 10**trial.suggest_int('min_delta', -8, -2)
    best_loss = float('inf') 
    early_stop_counter = 0
    best_weights = None
    start = time()
    for epoch in range(1000):
        # Forward pass
        outputs = skill(states)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

        #if (epoch+1) % 10 == 0:
        #    print(f'Epoch {epoch+1}: Loss = {loss.item():.8f}')
        
        # Check for early stopping
        if loss < best_loss - min_delta:
            best_loss = loss
            early_stop_counter = 0
            best_weights = skill.state_dict()
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} with loss {best_loss:.8f}')
            break
    end = time()
    #skill.load_state_dict(best_weights)
    return epoch, best_loss, end - start, sum(p.numel() for p in skill.parameters())

study = optuna.create_study(
    directions=['minimize', 'minimize', 'minimize', 'minimize'],
    storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
    study_name="overfit-nn-3"
    )
study.optimize(objective, n_trials=200)

print("Number of finished trials: ", len(study.trials))

