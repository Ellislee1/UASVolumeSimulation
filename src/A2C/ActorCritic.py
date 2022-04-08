from os import path

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims = 256, fc2_dims=256, chkpt_dir = 'tmp/ppo', file_prefix=""):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = path.join(chkpt_dir, f'{file_prefix}_actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def saveCheckpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def loadCheckpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims = 256,chkpt_dir = 'tmp/ppo', file_prefix=""):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = path.join(chkpt_dir, f'{file_prefix}critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def saveCheckpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def loadCheckpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
