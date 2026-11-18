import math
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .neural_network import QNetwork


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Batch(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class Transition(NamedTuple):
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
            raise TypeError("capacity deve ser um inteiro.")
        if capacity <= 0:
            raise ValueError("capacity deve ser maior que zero.")
        if isinstance(obs_size, bool) or not isinstance(obs_size, (int, np.integer)):
            raise TypeError("obs_size deve ser um inteiro.")
        if obs_size <= 0:
            raise ValueError("obs_size deve ser maior que zero.")

        self.capacity = int(capacity)
        self.obs_size = int(obs_size)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.buffer: list[Transition] = []
        self._next_idx = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if obs.shape != (self.obs_size,) or next_obs.shape != (self.obs_size,):
            raise ValueError("Dimensão dos vetores de estado diferente do esperado.")

        transition = Transition(
            obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ):
            raise TypeError("batch_size deve ser um inteiro.")
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Batch size inválido.")

        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        transitions = [self.buffer[index] for index in indices]
        return Batch(
            obs=np.asarray([item.obs for item in transitions], dtype=np.float32),
            actions=np.asarray([item.action for item in transitions], dtype=np.int64),
            rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
            next_obs=np.asarray(
                [item.next_obs for item in transitions], dtype=np.float32
            ),
            dones=np.asarray([item.done for item in transitions], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SelectionClientAgent:
    """Agente DQN para ambientes Gymnasium com observações vetoriais."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        eps_start: float = 0.9,
        eps_end: float = 0.01,
        eps_decay: float = 2500,
        tau: float = 0.005,
        batch_size: int = 128,
        gamma: float = 0.99,
        memory_size: int = 10000,
        hidden_sizes: tuple[int, ...] | list[int] = (128, 128),
        seed: int | None = None,
    ) -> None:
        if not hasattr(env.action_space, "n"):
            raise TypeError("O ambiente deve possuir um espaço de ações discreto.")

        self.env = env
        self.state_size = int(np.prod(env.observation_space.shape))
        if self.state_size <= 0:
            raise ValueError("O ambiente deve possuir observações vetoriais.")
        self.action_size = int(env.action_space.n)
        self.steps_done = 0

        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser maior que zero.")
        if not 0 <= eps_end <= eps_start <= 1:
            raise ValueError("É necessário ter 0 <= eps_end <= eps_start <= 1.")
        if eps_decay <= 0:
            raise ValueError("eps_decay deve ser maior que zero.")
        if not 0 < tau <= 1:
            raise ValueError("tau deve estar no intervalo (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma deve estar no intervalo [0, 1].")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")
        if memory_size <= 0:
            raise ValueError("memory_size deve ser maior que zero.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes deve conter inteiros positivos.")

        self.learning_rate = learning_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.rng = np.random.default_rng(seed)

        self.episode_durations: list[int] = []
        self.episode_rewards: list[float] = []
        self.losses: list[float] = []
        self.eps_history: list[float] = []
        self.wins: list[int] = []
        self.wins_rates: list[float] = []

        self.policy_net = QNetwork(
            self.state_size, self.action_size, list(hidden_sizes)
        ).to(device)
        self.target_net = QNetwork(
            self.state_size, self.action_size, list(hidden_sizes)
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True
        )
        self.memory = ReplayBuffer(
            self.memory_size, self.state_size, np.random.default_rng(seed)
        )

    def _state_tensor(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)

        if state_array.shape != (self.state_size,):
            raise ValueError("Dimensão do estado diferente da esperada.")

        return torch.as_tensor(state_array, dtype=torch.float32, device=device).unsqueeze(
            0
        )

    def get_action(self, state: np.ndarray | torch.Tensor, explore: bool = True) -> int:
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay
        )
        if explore:
            self.eps_history.append(epsilon)
            self.steps_done += 1
            if self.rng.random() < epsilon:
                return int(self.rng.integers(self.action_size))

        with torch.no_grad():
            return int(self.policy_net(self._state_tensor(state)).argmax(dim=1).item())

    def optimize_model(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)

        state_action_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(
            1
        )
        with torch.no_grad():
            next_state_values = self.target_net(next_obs).max(dim=1).values
            expected_values = rewards + self.gamma * next_state_values * (1 - dones)

        loss = F.smooth_l1_loss(state_action_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        loss_value = float(loss.item())
        self.losses.append(loss_value)
        return loss_value

    def update_target_network(self) -> None:
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.lerp_(policy_param, self.tau)

    def test(self, num_episodes: int = 100) -> dict[str, float]:
        if num_episodes <= 0:
            raise ValueError("num_episodes deve ser maior que zero.")

        was_training = self.policy_net.training
        self.policy_net.eval()
        rewards: list[float] = []
        try:
            for _ in range(num_episodes):
                state, _ = self.env.reset()
                episode_over = False
                total_reward = 0.0

                while not episode_over:
                    action = self.get_action(state, explore=False)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    episode_over = terminated or truncated
                    total_reward += float(reward)
                rewards.append(total_reward)
                
        finally:
            self.policy_net.train(was_training)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
        }


def train(
    agent: SelectionClientAgent,
    num_episodes: int = 500,
    seed: int | None = None,
    win_threshold: float = 500,
) -> SelectionClientAgent:
    """Treina o agente DQN e retorna o agente com seu histórico atualizado."""
    if num_episodes <= 0:
        raise ValueError("num_episodes deve ser maior que zero.")

    if seed is not None:
        agent.rng = np.random.default_rng(seed)
        agent.memory.rng = np.random.default_rng(seed)
        agent.env.action_space.seed(seed)

    agent.policy_net.train()
    for episode in range(num_episodes):
        state, _ = agent.env.reset(seed=None if seed is None else seed + episode)
        episode_over = False
        total_reward = 0.0
        steps = 0

        while not episode_over:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            episode_over = terminated or truncated
            agent.memory.add(
                state,
                action,
                reward,
                next_state,
                done=terminated,
            )
            agent.optimize_model()
            agent.update_target_network()

            state = next_state
            total_reward += float(reward)
            steps += 1

        agent.episode_durations.append(steps)
        agent.episode_rewards.append(total_reward)
        agent.wins.append(int(total_reward >= win_threshold))
        agent.wins_rates.append(float(np.mean(agent.wins[-50:])))

    return agent
