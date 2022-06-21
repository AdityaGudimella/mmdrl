import typing as t

import torch
import numpy as np


class ParticleDQNType(t.NamedTuple):
    patricles: torch.Tensor

    @property
    def q_values(self) -> torch.Tensor:
        return torch.mean(self.patricles, dim=2)


class ParticleDQNet(torch.nn.Module):
    def __init__(self, num_actions: int, num_atoms: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, (8, 8), stride=4, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (4, 4), stride=2, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), stride=1, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            output_size = self.conv(torch.zeros(1, 4, 84, 84)).shape[1]
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(output_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions * num_atoms),
        )

    def forward(self, state: torch.Tensor) -> ParticleDQNType:
        states = state.float().div(255)
        particles = self.dense(self.conv(states)).view(-1, self.num_actions, self.num_atoms)
        return ParticleDQNType(particles)
