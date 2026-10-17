"""pytorchexample: A Flower / PyTorch app."""
import numpy as np
import torch


def _divergence_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a divergência KL entre duas distribuições de probabilidade p e q.

    Parâmetros:
        - p: distribuição de probabilidade p
        - q: Distribuição de probabilidade q

    Retorna:
        divergência KL D_KL(p || q)
    """

    # Adiciona um pequeno valor para evitar log(0)
    epsilon = 1e-10
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)

    return np.sum(p_safe * np.log(p_safe / q_safe))


def _compute_class_proportions(labels, num_classes: int = 10) -> np.ndarray:
    """Calcula a proporção P_i(c) de cada classe c em uma partição de dados do
    cliente i.

    Parâmetros:
        - labels: Lista, np.ndarray ou torch.Tensor contendo os rótulos do
        cliente
        - num_classes: número total de classes do dataset

    Retorna:
        - array de tamanho (num_classes,) com as proporções normalizadas
        P_i
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    elif isinstance(labels, list):
        labels = np.array(labels)

    # Caso a partição esteja vazia
    if len(labels) == 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    # Conta a frequência de cada classe de 0 até num_classes - 1
    counts = np.bincount(labels, minlength=num_classes)

    # Converte para frequências relativas (proporções)
    proportions = counts / len(labels)

    return proportions.astype(np.float32)


def _compute_proportions_for_clients(client_labels: list[np.ndarray], num_classes: int = 10) -> np.ndarray:
    """Calcula as proporções P_i(c) para cada cliente i.

    Parâmetros:
        - client_labels: lista de rótulos de cada cliente
        - num_classes: número total de classes do dataset

    Retorna:
        - matriz de proporções P_i(c) para cada cliente
    """
    proportions = np.array(
        [_compute_class_proportions(labels, num_classes) for labels in client_labels],
        dtype=np.float32,
    )
    return proportions


def compute_divergence_kl_for_clients(
    client_labels: list[np.ndarray], global_labels: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Calcula a divergência KL D_KL(P_i || P_global) para cada cliente i.

    Parâmetros:
        - client_labels: lista de rótulos de cada cliente
        - global_labels: rótulos globais
        - num_classes: número total de classes do dataset

    Retorna:
        - vetor de divergências KL para cada cliente
    """

    global_proportions = _compute_class_proportions(global_labels, num_classes)
    client_proportions = _compute_proportions_for_clients(client_labels, num_classes)

    divergences = np.array(
        [
            _divergence_kl(client_prop, global_proportions)
            for client_prop in client_proportions
        ],
        dtype=np.float32,
    )
    return divergences


def _generate_bandwidths(
    num_clients: int,
    min_mbps: float = 1.0,
    max_mbps: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """Gera a banda larga B_i para N clientes heterogêneos.

    Parâmetros:
        - num_clients: quantidade de clientes
        - min_mbps: número mínimo de megabits por segundo
        - max_mbps: número máximo de megabits por segundo
        - seed: valor da seed

    Retorna:
        - valores das banda largas para os N clientes, em Bytes/segundo (B/s).
    """

    np.random.seed(seed)
    # Sorteia a banda em Megabits por segundo (Mbps)
    bandwidth_mbps = np.random.uniform(min_mbps, max_mbps, size=num_clients)

    # Converte Mbps -> Bytes por segundo: (Mbps * 1.000.000) / 8
    bandwidth_bytes_per_sec = (bandwidth_mbps * 1e6) / 8.0

    return bandwidth_bytes_per_sec.astype(np.float32)


def _generate_latencies(
    num_clients: int,
    min_ms: float = 10.0,
    max_ms: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Gera a latência de rede delta_i para N clientes.

    Parâmetros:
        - num_clients: quantidade de clientes
        - min_mbps: número mínimo de milissegundos
        - max_mbps: número máximo de milissegundos
        - seed: valor da seed

    Retorna:
        - os valores de latências para os N clientes, em segundos.
    """

    np.random.seed(seed + 100)  # Semente distinta para não alinhar com a banda
    # Sorteia a latência em milissegundos (ms)
    latencies_ms = np.random.uniform(min_ms, max_ms, size=num_clients)

    # Converte ms -> segundos
    latencies_sec = latencies_ms / 1000.0

    return latencies_sec.astype(np.float32)


def compute_model_size_bytes(model: torch.nn.Module) -> int:
    """Calcula o tamanho total em bytes do modelo neural PyTorch (pesos +
    biases).

    Parâmetros: 
        - model: modelo de rede neural

    Retorna:
        - o valor em bytes do tamanho do modelo
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _compute_communication_costs(
    model_size_bytes: float,
    bandwidths_bytes_per_sec: np.ndarray,
    latencies_sec: np.ndarray,
) -> np.ndarray:
    """Calcula o custo c_i (em segundos) para cada cliente conforme a fórmula
    do MDP: c_i = (|theta| / B_i) + delta_i.

    Parâmetros:
        - model_size_bytes: tamanho do modelo em bytes
        - bandwidths_bytes_per_sec: valores das bandas largas dos clientes
        - latencies_sec: valores das latências dos clientes

    Retorna:
        - os custos de comunicação para cada cliente.
    """
    costs_sec =  (model_size_bytes / bandwidths_bytes_per_sec) + latencies_sec

    return costs_sec.astype(np.float32)


def generate_client_communication_costs(
    model: torch.nn.Module,
    num_clients: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Função wrapper para gerar diretamente o vetor de custo de comunicação para
    todos clientes.

    Parâmetros:
        - model: modelo de rede neural
        - num_clients: número de clientes
        - seed: valor da seed

    Retorna:
        - vetor de custos de comunicação, com valor para cada cliente.
    """
    model_bytes = compute_model_size_bytes(model)
    bandwidths = _generate_bandwidths(num_clients, seed=seed)
    latencies = _generate_latencies(num_clients, seed=seed)

    costs = _compute_communication_costs(model_bytes, bandwidths, latencies)
    return costs