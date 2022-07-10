import random
import typing as t
import collections

import numpy as np
import numpy.typing as npt
import torch


ReplayElement = collections.namedtuple("shape_type", ["name", "shape", "type"])
NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = torch.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

MMDOutputs = collections.namedtuple("MMDOutputs", ["particles", "q_values"])


def gaussian_rbf_kernel(d: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """

    :param d: shape: (batch_size, num_samples, num_samples)
    """
    b, n, n = d.shape
    sigmas_tensor = torch.as_tensor(sigmas, dtype=torch.float32, device=d.device)
    k = sigmas_tensor.size(0)
    h = 1 / sigmas_tensor.view(-1, 1)
    s = h @ d.view(1, -1)
    assert s.shape == (k, b*n*n)
    return torch.exp(-s).sum(dim=0).view(d.shape)


def huber_loss(u: torch.Tensor, kappa: float = 1) -> torch.Tensor:
    if kappa == 0:
        return u.abs()
    huber_loss_case_one = u.abs().le(kappa).float() * 0.5 * u ** 2
    huber_loss_case_two = u.abs().gt(kappa).float() * kappa * (u.abs() - 0.5 * kappa)
    return huber_loss_case_one + huber_loss_case_two




class NatureDQNNetwork(torch.nn.Module):
    def __init__(self, num_actions: int) -> None:
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=NATURE_DQN_STACK_SIZE,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x.float().div(255))


def linearly_decaying_epsilon(
    decay_period: float, step: int, warmup_steps: int, epsilon: float
):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1 - epsilon)
    return epsilon + bonus


def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps, epsilon):
    return epsilon


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        num_atoms: int,
        observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
        observation_dtype=NATURE_DQN_DTYPE,
        stack_size=NATURE_DQN_STACK_SIZE,
        network=NatureDQNNetwork,
        batch_size: int = 32,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=20000,
        update_period=4,
        target_update_period=8000,
        epsilon_fn=linearly_decaying_epsilon,
        epsilon_train=0.01,
        epsilon_eval=0.001,
        epsilon_decay_period=250000,
        device="cpu",
        eval_mode=False,
        optimizer=torch.optim.RMSprop,
    ) -> None:
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.network = network
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = gamma**update_horizon
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.update_period = update_period
        self.eval_mode = eval_mode
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((1, *self.observation_shape, stack_size), dtype=self.observation_dtype)
        self._replay = self._build_replay_buffer()
        self._build_networks()

    def begin_episode(self, observation: npt.NDArray[t.Any]) -> int:
        self._reset_state()
        self._record_observation(observation)
        self.action = self._select_action()
        return self.action

    def step(self, reward: float, observation: npt.NDArray[t.Any]) -> int:
        self._last_observation = self._observation
        self._record_observation(observation)
        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False)
            self._train_step()

        self.action = self._select_action()
        return self.action

    def end_episode(self, reward: float) -> None:
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward, True)

    def _select_action(self):
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(
                self.epsilon_decay_period,
                self.training_steps,
                self.min_replay_history,
                self.epsilon_train,
            )
        if random.random() <= epsilon:
            return random.randint(0, self.num_actions - 1)
        return self.online_convnet(torch.as_tensor(self.states, device=self.device)).argmax(dim=1).item()

    def _record_observation(self, observation: npt.NDArray[t.Any]) -> None:
        self._observation = np.reshape(observation, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1] = self._observation

    def _reset_state(self) -> None:
        self.state.fill(0)

    def _build_networks(self):
        self.online_convnet = self.network(self.num_actions).to(self.device)
        self.target_convnet = self.network(self.num_actions).to(self.device)
        self.target_convnet.load_state_dict(self.online_convnet.state_dict())
        self.target_convnet.requires_grad_(False)

    def _build_replay_buffer(self):
        pass

    def _store_transition(
        self, last_observation, action, reward, is_terminal, priority=None
    ) -> None:
        if priority is None:
            if self._replay_scheme == "uniform":
                priority = 1
            else:
                raise ValueError(f"Unrecognized replay scheme: {self._replay_scheme}")
        if not self.eval_mode:
            self._replay.add(
                last_observation,
                action,
                reward,
                is_terminal,
                action,
                reward,
                is_terminal,
                priority,
            )

    def _update_target_net(self) -> None:
        self.target_convnet.load_state_dict(self.online_convnet)

    def _target_particles(
        self,
        replay_next_states: torch.Tensor,
        replay_rewards: torch.Tensor,
        replay_terminals: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            rewards = replay_rewards.view(-1, 1, 1)
            non_terminals = 1 - replay_terminals.float()
            gamma_with_terminal = self.cumulative_gamma * non_terminals.view(-1, 1, 1)
            replay_next_target_net_outputs = self.target_convnet(replay_next_states)
            next_particles = replay_next_target_net_outputs.particles
            target_particles = rewards + gamma_with_terminal * next_particles
            return self._action_sampler.compute_target(
                target_particles, estimator=self._target_estimator
            )

    def _chosen_action_particles(
        self, replay_states: torch.Tensor, replay_actions: torch.Tensor
    ) -> torch.Tensor:
        net_outputs = self.online_convnet(replay_states)
        result = net_outputs.particles[torch.arange(self.batch_size), replay_actions]
        assert result.shape == (self.batch_size, self.num_atoms)
        return result

    def _calculate_loss(
        self,
        replay_states: torch.Tensor,
        replay_actions: torch.Tensor,
        replay_next_states: torch.Tensor,
        replay_rewards: torch.Tensor,
        replay_terminals: torch.Tensor,
    ) -> torch.Tensor:
        target_particles = self._target_particles(
            replay_next_states, replay_rewards, replay_terminals
        )
        chosen_action_particles = self._chosen_action_particles(replay_states, replay_actions)

        diff1 = chosen_action_particles[:, :, None] - chosen_action_particles[:,None,:]
        diff2 = chosen_action_particles[:,:, None] - target_particles[:,None,:]
        diff3 = target_particles[:,:, None] - target_particles[:,None,:]
        if self.kappa == 0:
            d1 = torch.square(diff1)
            d2 = torch.square(diff2)
            d3 = torch.square(diff3)
        else:
            d1 = huber_loss(diff1, kappa=self.kappa)
            d2 = huber_loss(diff2, kappa=self.kappa)
            d3 = huber_loss(diff3, kappa=self.kappa)
        assert d1.shape == d2.shape == d3.shape == (self.batch_size, self.num_atoms, self.num_atoms)

        assert self.bandwidth_selection_type == "mixture"
        sigmas = list(range(1, 11))
        xixj = gaussian_rbf_kernel(d1, sigmas).mean(dim=-1).mean(dim=-1)
        xiyj = gaussian_rbf_kernel(d2, sigmas).mean(dim=-1).mean(dim=-1)
        yiyj = gaussian_rbf_kernel(d3, sigmas).mean(dim=-1).mean(dim=-1)

        mmd_squared = xixj + yiyj - 2 * xiyj
        mmd_squared = mmd_squared.clamp(min=0)
        return mmd_squared.mean()

    def _train_step(self) -> None:
        if self._replay.memory.add_count <= self.min_replay_history:
            return
        if self.training_steps % self.update_period == 0:
            samples = self._replay.sample()
            loss = self._calculate_loss(
                replay_states=samples.states,
                replay_actions=samples.actions,
                replay_next_states=samples.next_states,
                replay_rewards=samples.rewards,
                replay_terminals=samples.terminals,
            )
        if self.training_steps % self.target_update_period == 0:
            self._update_target_net()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
