"""pytorchexample: A Flower / PyTorch app."""
import numpy as np
import torch


def _divergence_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula a divergência KL entre duas distribuições de probabilidade p e q.

    Args:
        p (np.ndarray): Distribuição de probabilidade p.
        q (np.ndarray): Distribuição de probabilidade q.

    Returns:
        float: Divergência KL D_KL(p || q).
    """
    # Adiciona um pequeno valor para evitar log(0)
    epsilon = 1e-10
    p_safe = np.clip(p, epsilon, 1)
    q_safe = np.clip(q, epsilon, 1)

    return np.sum(p_safe * np.log(p_safe / q_safe))


def _compute_class_proportions(labels, num_classes: int = 10) -> np.ndarray:
    """Calcula a proporção P_i(c) de cada classe c em uma partição de dados do
    cliente i.

    Parametros:
        labels: Lista, np.ndarray ou torch.Tensor contendo os rótulos do
        cliente.
        num_classes: Número total de classes do dataset (padrão: 10 para
        CIFAR-10).

    Retorna:
        np.ndarray de tamanho (num_classes,) com as proporções normalizadas
        P_i.
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

    Args:
        client_labels (list[np.ndarray]): Lista de rótulos de cada cliente.
        num_classes (int): Número total de classes do dataset.

    Returns:
        np.ndarray: Matriz de proporções P_i(c) para cada cliente.
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

    Args:
        client_labels (list[np.ndarray]): Lista de rótulos de cada cliente.
        global_labels (np.ndarray): Rótulos globais.
        num_classes (int): Número total de classes do dataset.

    Returns:
        np.ndarray: Vetor de divergências KL para cada cliente.
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

    Retorna os valores convertidos para Bytes/segundo (B/s).
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

    Retorna os valores convertidos para segundos.
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
    """
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _compute_communication_costs(
    model_size_bytes: float,
    bandwidths_bytes_per_sec: np.ndarray,
    latencies_sec: np.ndarray,
) -> np.ndarray:
    """Calcula o custo c_i (em segundos) para cada cliente conforme a fórmula

    do MDP:

    c_i = (|theta| / B_i) + delta_i
    """
    costs_sec =  (model_size_bytes / bandwidths_bytes_per_sec) + latencies_sec
    return costs_sec.astype(np.float32)


def generate_client_communication_costs(
    model: torch.nn.Module,
    num_clients: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Função wrapper para gerar diretamente o vetor c_i de tamanho

    (num_clients,)
    """
    model_bytes = compute_model_size_bytes(model)
    bandwidths = _generate_bandwidths(num_clients, seed=seed)
    latencies = _generate_latencies(num_clients, seed=seed)

    costs = _compute_communication_costs(model_bytes, bandwidths, latencies)
    return costs