import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..utils.common import set_seed

class SimulationEnvironment(gym.Env):
    """ Ambiente gymnasium de simulação federada. """
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

        # Componentes do ambiente
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

        # Atributos do ambiente gymnasium
        self.observation_space = spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=(num_clients, 6),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=(num_clients,),
            dtype=np.float32,
        )

    @property
    def eligible_clients_mask(self) -> np.ndarray:
        """
        Retorna o vetor de disponibilidade de cada cliente na rodada atual.
        """
        return self.availability_simulator.current_inst_avail.copy()

    def reset(
        self,
        seed: int = 42,
    ):
        """
        Recomeço do ambiente federado, com métricas zeradas.

        Parâmetros: 
            - seed: valor da seed

        Retorna:
            - obs: estado de observação do ambeinte
            - info: informações adicionais, com máscara de clientes elegíveis
        """
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

    def step(
        self,
        loss: float,
        accuracy: float,
        clients_selected: np.ndarray,
        client_losses: np.ndarray,
    ):
        """
        Passo no ambiente simulado: recebe a ação (clientes selecionados) e atualiza
        o estado do MDP.

        Parâmetros:
            - loss: loss global do modelo
            - accuracy: acurácia global do modelo
            - clients_selected: clientes selecionados para o treinamento
            - client_losses: vetor com as perdas dos clientes selecionados

        Retorna:
            - next_obs: novo estado do MDP, após realizar a ação
            - reward: recompensa obtida pela ação realizada
            - done: True se atingiu o número máximo de rodadas, ou False
            caso contrário
            - truncated: sempre retorna False
            - info: informações adicionais do passo realizado, com
            rodada, bytes comunicados, melhor acurácia global, última
            acurácia, index de Jain e máscara de clientes selecionados.
        """
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
            current_accuracy=accuracy,
            selected_clients=clients_selected,
            w1=10.0,
            w2=0.5,
            w3=1.0
        )

        self.last_loss = loss
        self.last_accuracy = accuracy
        self.best_global_accuracy = max(self.best_global_accuracy, accuracy)
        self.round += 1

        # Verifica condições de término
        done = self.round >= self.max_rounds
        truncated = False

        num_clients_selected = np.sum(clients_selected)

        # Informações adicionais sobre o ambiente
        info = {
            "round": self.round,
            "total_communication_bytes": self.total_communication_bytes,
            "best_global_accuracy": self.best_global_accuracy,
            "last_accuracy": self.last_accuracy,
            "jain_fairness_index": self._compute_jain_fairness(),
            "eligible_clients_mask": next_eligible_clients,
            "number_clients_selected": num_clients_selected
        }

        return next_obs, reward, done, truncated, info

    def get_total_communication_bytes(self) -> int:
        """Retorna o total de bytes comunicados por mensagens até então."""
        return self.total_communication_bytes

    def record_communication_bytes(self, num_bytes: int) -> None:
        """
        Adiciona bytes de comunicação ao total de bytes comunicados até então.
        
        Parâmetros:
            - num_bytes: quantidade de bytes comunicados na rodada.
        """
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        
        self.total_communication_bytes += int(num_bytes)

    def get_final_metrics(self) -> tuple[float, float, int]:
        """
        Retorna as métricas finais.
        
        Retorna:
            - melhor acurácia global
            - última loss global
            - total de bytes comunicados até então
        """
        return self.best_global_accuracy, self.last_loss, self.total_communication_bytes

    def _reward_communication(self, selected_clients: np.ndarray) -> float:
        """
        Cálculo da recompensa de comunicação normalizada.

        Parâmetros:
            - selected_clients: clientes selecionados na rodada

        Retorna:
            - recompensa de comunicação normalizada
        """
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
        """
        Computa o índice de Jain de equidade.

        Retorna:
            - o valor do índice de Jain da rodada.
        """
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

    def compute_reward(
        self,
        current_accuracy: float,
        selected_clients: np.ndarray,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        beta: float = 3.0
    ) -> float:
        """
        Computa a recompensa da rodada, baseado na acurácia, bytes de comunicação
        e índice de jain.

        Parâmetros:
            - current_accuracy: acurácia global da última rodada
            - selected_clients: máscara de clientes selecionados da última rodada
            - w1: peso da recompensa de acurácia
            - w2: peso da recompensa de comunicação
            - w3: peso da recompensa do índice de Jain
            - beta: peso para diferença negativa de acurácia

        Retorna:
            - recompensa total da rodada
        """
        reward_acc = current_accuracy - self.last_accuracy if self.round > 0 else 0.0
        if reward_acc < 0:
            reward_acc = reward_acc * beta

        reward_comm = self._reward_communication(selected_clients)
        reward_fair = self._compute_jain_fairness()

        return w1 * reward_acc + w2 * reward_comm + w3 * reward_fair

class AvailabilitySimulator:
    """ MDP de disponibilidade dos clientes em cada rodada. """

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

    def reset(self, seed: int = 42) -> np.ndarray:
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
        """Avança a cadeia de Markov de 2 estados em 1 rodada.

        Retorna:
            - lista/array binário de clientes fisicamente disponíveis nesta
            rodada
            - a média móvel a_i.
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
    """ Estado do MDP. """
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

        Retorna:
            - a matriz de observação S(0).
        """
        self.S[:, 0] = 0.0  # Loss inicial
        self.S[:, 3] = 1.0  #  Média móvel inicial
        self.S[:, 5] = 0.0  # Tau_i inicial

        return self.S

    def get_observation_matrix(self) -> np.ndarray:
        """Retorna a matriz de observação atual S."""
        return self.S

    def step(
        self,
        selected_mask: np.ndarray,
        losses: np.ndarray,
        mean_availabilities: np.ndarray,
    ):
        """
        Realiza passo no estado no MDP, atualizando a loss, o tempo 
        desde a última seleção de cada cliente e a média de disponibilidade.

        Parâmetros:
            - selected_mask: máscara de clientes selecionados
            - losses: perdas dos clientes selecionados na rodada
            - mean_availabilities: médias de disponibilidade dos clientes
        """
        # Atualiza a loss dos selecionados
        self.S[selected_mask, 0] = losses[selected_mask]

        # Atualiza o tempo desde a última seleção tau_i
        self.S[selected_mask, 5] = 0.0
        self.S[~selected_mask, 5] += 1.0

        # Atualiza a disponibilidade média a_i
        self.S[:, 3] = mean_availabilities
