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

    positive_classes = p > 0
    if np.any(q[positive_classes] == 0):
        return float("inf")

    divergence = np.sum(
        p[positive_classes]
        * np.log(p[positive_classes] / q[positive_classes])
    )
    return float(max(divergence, 0.0))


def _compute_class_counts(labels, num_classes: int) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    labels = labels.reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot compute class proportions from empty labels.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Labels must contain integer class indices.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"Labels must be in the range [0, {num_classes - 1}]."
        )

    return np.bincount(labels, minlength=num_classes)


def compute_divergence_kl_for_clients(
    client_labels: list[np.ndarray], num_classes: int = 10
) -> np.ndarray:
    """Calcula a divergência KL D_KL(P_i || P_global) para cada cliente i.

    Parâmetros:
        - client_labels: lista de rótulos de cada cliente
        - num_classes: número total de classes do dataset

    A distribuição global é calculada pela soma das contagens de todas as
    partições, ponderando cada cliente pelo número de exemplos.

    Retorna:
        - vetor de divergências KL para cada cliente
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")
    if not client_labels:
        raise ValueError("At least one client partition is required.")

    client_counts = np.stack(
        [
            _compute_class_counts(labels, num_classes)
            for labels in client_labels
        ]
    )
    global_counts = client_counts.sum(axis=0)
    global_proportions = global_counts / global_counts.sum()
    client_proportions = client_counts / client_counts.sum(axis=1, keepdims=True)

    divergences = np.array(
        [
            _divergence_kl(client_prop, global_proportions)
            for client_prop in client_proportions
        ],
        dtype=np.float64,
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