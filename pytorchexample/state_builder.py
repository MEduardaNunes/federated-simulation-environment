import math

import numpy as np


class SimulationEnvironment:

    def __init__(
        self,
        num_clients: int,
        divergences_kl: np.ndarray,
        number_of_examples: np.ndarray,
        costs_communication: np.ndarray,
        max_rounds: int = 200,
        k_window: int = 10,
        p_on_on: float = 0.8,
        p_off_on: float = 0.2,
        seed: int = 42,
        rounds_warm_up: int = 20
    ):

        # Atributos do ambiente de simulação
        self.num_clients = num_clients
        self.max_rounds = max_rounds

        # Métricas de rastreamento do ambiente
        self.round = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0
        self.best_global_accuracy = 0.0
        self.total_communication_bytes = 0
        self.participation_count = np.zeros(num_clients, dtype=int)
        self.min_reward_communication_bytes = math.inf
        self.max_reward_communication_bytes = -math.inf
        self.rounds_warm_up = rounds_warm_up

        self.availability_simulator = AvailabilitySimulator(
            num_clients=num_clients,
            k_window=k_window,
            p_on_on=p_on_on,
            p_off_on=p_off_on,
            seed=seed,
        )

        self.state_builder = MDPStateBuilder(
            num_clients=num_clients,
            divergences_kl=divergences_kl,
            number_of_examples=number_of_examples,
            costs_communication=costs_communication,
        )

    def reset(
        self,
        seed: int = 42,
    ):
        self.round = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0
        self.best_global_accuracy = 0.0
        self.total_communication_bytes = 0
        self.participation_count.fill(0)

        eligible_clients = self.availability_simulator.reset(seed=seed)
        obs = self.state_builder.reset()

        info = {"eligible_clients_mask": eligible_clients}

        return obs, info

    @property
    def eligible_clients_mask(self) -> np.ndarray:
        """Return the physical availability of each client in the current round."""
        return self.availability_simulator.current_inst_avail.copy()

    def step(
        self,
        last_loss: float,
        accuracy: float,
        clients_selected: np.ndarray,
        client_losses: np.ndarray,
    ):
        clients_selected = np.asarray(clients_selected, dtype=bool)
        client_losses = np.asarray(client_losses, dtype=np.float32)

        self.participation_count[clients_selected] += 1

        # Atualiza o estado do MDP com as perdas dos clientes selecionados e a disponibilidade média
        next_eligible_clients, mean_availabilities = self.availability_simulator.step()

        self.state_builder.step(
            selected_mask=clients_selected,
            losses=client_losses,
            mean_availabilities=mean_availabilities,
        )

        next_obs = self.state_builder.get_observation_matrix()

        # Calcula recompensa
        reward = self.compute_reward(
            current_loss=last_loss,
            selected_clients=clients_selected,
            w1=1.0,
            w2=1.0,
            w3=1.0
        )

        self.last_loss = last_loss
        self.last_accuracy = accuracy
        self.best_global_accuracy = max(self.best_global_accuracy, accuracy)
        self.round += 1

        # Verifica condições de término
        done = self.round >= self.max_rounds
        truncated = False

        # Informações adicionais sobre o ambiente
        info = {
            "round": self.round,
            "total_communication_bytes": self.total_communication_bytes,
            "best_global_accuracy": self.best_global_accuracy,
            "last_accuracy": self.last_accuracy,
            "jain_fairness_index": self._compute_jain_fairness(),
            "eligible_clients_mask": next_eligible_clients,
        }

        return next_obs, reward, done, truncated, info

    def get_total_communication_bytes(self) -> int:
        """Return bytes counted in Flower record payloads sent and received."""
        return self.total_communication_bytes

    def record_communication_bytes(self, num_bytes: int) -> None:
        """Add Flower record payload bytes, excluding message/transport overhead."""
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        self.total_communication_bytes += int(num_bytes)

    def get_final_metrics(self) -> tuple[float, float, int]:
        """Retorna a melhor acurácia global alcançada até o momento."""
        return self.best_global_accuracy, self.last_loss, self.total_communication_bytes

    def _reward_communication(self, selected_clients: np.ndarray) -> float:
        if np.sum(selected_clients) == 0:
            return 0.0

        observation_matrix = self.state_builder.get_observation_matrix()
        comm_costs = observation_matrix[selected_clients, 4]  # c_i for selected
        sum_comm = -np.sum(comm_costs)

        self.min_reward_communication_bytes = min(self.min_reward_communication_bytes, sum_comm)
        self.max_reward_communication_bytes = max(self.max_reward_communication_bytes, sum_comm)

        if self.round < self.rounds_warm_up or self.max_reward_communication_bytes == self.min_reward_communication_bytes:
            return 0.5

        normalized_comm_reward = (sum_comm - self.min_reward_communication_bytes) / (self.max_reward_communication_bytes - self.min_reward_communication_bytes)

        return normalized_comm_reward

    def _compute_jain_fairness(self) -> float:
        """Compute Jain's fairness index based on the number of times each client participated."""
        if self.num_clients <= 0:
            return 1.0

        participation = self.participation_count.astype(np.float64, copy=False)
        total_participation = float(np.sum(participation))

        if total_participation <= 0.0:
            return 1.0

        sum_participation_squared = float(np.sum(participation ** 2))
        if np.isclose(sum_participation_squared, 0.0):
            return 1.0

        fairness_index = (total_participation ** 2) / (self.num_clients * sum_participation_squared)
        return float(np.clip(fairness_index, 0.0, 1.0))

    def compute_reward(self, 
        current_loss: float,
        selected_clients: np.ndarray,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0
    ) -> float:
        # Placeholder for reward computation logic
        reward_acc = current_loss - self.last_loss
        reward_comm = self._reward_communication(selected_clients)
        reward_fair = self._compute_jain_fairness()

        return w1 * reward_acc +  w2 * reward_comm + w3 * reward_fair

class AvailabilitySimulator:
    def __init__(
        self,
        num_clients: int,
        k_window: int = 10,
        p_on_on: float = 0.8,
        p_off_on: float = 0.2,
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.k_window = k_window
        self.seed = seed
        np.random.seed(seed)

        # Parâmetros da dinâmica de disponibilidade
        self.p_on_on = p_on_on
        self.p_off_on = p_off_on

        # Disponibilidade física instantânea atual (vetor binário N)
        self.current_inst_avail = np.ones(num_clients, dtype=bool)

        # Histórico das últimas k rodadas (matriz N x k) para calcular a média móvel
        self.availability_history = np.ones(
            (num_clients, k_window), dtype=bool
        )

    def reset(self, seed: int = 42) -> tuple[np.ndarray]:
        """Reseta o simulador de disponibilidade para o estado inicial.

        Retorna:
            - eligible_clients: vetor binário indicando quais clientes estão
              fisicamente disponíveis na rodada inicial.
        """
        self.seed = seed
        np.random.seed(seed)

        # Inicializa todos os clientes como disponíveis
        self.current_inst_avail = np.ones(self.num_clients, dtype=bool)
        self.availability_history = np.ones(
            (self.num_clients, self.k_window), dtype=bool
        )

        return self.current_inst_avail

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Avança a cadeia de Markov de 2 estados em 1 rodada e atualiza a

        coluna 3 (a_i).

        Retorna a lista/array binário de clientes fisicamente disponíveis nesta
        rodada e a média móvel a_i.
        """
        # Sorteia números aleatórios uniformes para cada cliente
        rand_vals = np.random.rand(self.num_clients)

        # Transição da Cadeia de Markov:
        # Se estava ON (True): permanece ON se rand_val < p_on_on
        # Se estava OFF (False): passa para ON se rand_val < p_off_on
        new_inst_avail = np.where(
            self.current_inst_avail,
            rand_vals < self.p_on_on,
            rand_vals < self.p_off_on,
        )

        self.current_inst_avail = new_inst_avail

        # Atualiza a janela móvel: desloca as colunas para a esquerda e insere a nova disponibilidade na última posição
        self.availability_history[:, :-1] = self.availability_history[:, 1:]
        self.availability_history[:, -1] = new_inst_avail

        # Calcula a média móvel a_i^(t)
        a_i = np.mean(self.availability_history, axis=1)

        # Retorna o vetor booleano instantâneo para filtrar os clientes elegíveis A(t)
        return self.current_inst_avail, a_i


class MDPStateBuilder:
    def __init__(
        self,
        num_clients: int,
        divergences_kl: np.ndarray,
        number_of_examples: np.ndarray,
        costs_communication: np.ndarray,
    ):
        self.num_clients = num_clients
        # Colunas: [0: loss, 1: D_kl, 2: n_i, 3: a_i, 4: c_i, 5: tau_i]
        self.S = np.zeros((num_clients, 6), dtype=np.float32)

        # Atributos estáticos
        self.S[:, 1] = divergences_kl
        self.S[:, 2] = number_of_examples
        self.S[:, 3] = 1.0  # Média móvel inicial
        self.S[:, 4] = costs_communication
        

    def reset(self) -> np.ndarray:
        """Reseta a matriz de observação S(t) para o estado inicial.

        Retorna a matriz de observação S(0).
        """
        self.S[:, 0] = 0.0  # Loss inicial
        self.S[:, 3] = 1.0  #  Média móvel inicial
        self.S[:, 5] = 0.0  # Tau_i inicial

        return self.S

    def get_observation_matrix(self) -> np.ndarray:
        """Return the current per-client observation matrix."""
        return self.S

    def step(
        self,
        selected_mask: np.ndarray,
        losses: np.ndarray,
        mean_availabilities: np.ndarray,
    ):
        # Atualiza a loss dos selecionados
        self.S[selected_mask, 0] = losses[selected_mask]

        # Atualiza o tempo desde a última seleção tau_i
        self.S[selected_mask, 5] = 0.0
        self.S[~selected_mask, 5] += 1.0

        # Atualiza a disponibilidade média a_i
        self.S[:, 3] = mean_availabilities
