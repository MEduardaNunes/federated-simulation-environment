import csv
import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from structure.classes.federated_classes import SimulationEnvironment
from structure.classes.custom_strategy import SimulationFedAvg
from structure.classes.neural_network import Net
from structure.utils.dataset_config import DEFAULT_DATASET, get_dataset_config
from structure.utils.data import (
    load_centralized_dataset,
    partition_cache_dir,
    prepare_partitions,
)

from structure.utils.training import test
from structure.utils.common import set_seed
from structure.utils.environment_utils import (
    compute_divergence_kl_for_clients,
    generate_client_communication_costs,
)
from structure.utils.generate_plots import *

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"]="0"
set_seed(42)  # Define a semente para reprodutibilidade

# Create ServerApp
app = ServerApp()
round_metrics_path = Path.cwd() / "round_metrics.csv"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Entrada principal para o servidor. Coordena o treinamento
    global.

    Parâmetros:
        - grid: infraestrutura da rede dos nós do sistema Flower
        - context: contexto da simulação, contendo principais hiperparâmetros
        para o treinamento 
    """

    # Leitura da configuração de execução
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    num_partitions = len(list(grid.get_node_ids()))
    alpha: float = context.run_config["alpha-dirichlet"]
    K : int = context.run_config["K"]
    dataset = context.run_config.get("dataset", DEFAULT_DATASET)
    dataset_config = get_dataset_config(dataset)

    # Criação das partições em disco e leitura das labels dos dados de cada
    # cliente para cálculo de divergências KL
    prepare_partitions(num_partitions, alpha, dataset=dataset)
    client_labels = []
    number_of_examples = np.zeros(num_partitions, dtype=np.int64)
    for partition_id in range(num_partitions):
        partition = load_from_disk(
            str(
                partition_cache_dir(
                    num_partitions, alpha, dataset=dataset
                )
                / f"partition-{partition_id}"
            )
        )
        labels = np.asarray(partition["train"]["label"], dtype=np.int64)
        client_labels.append(labels)
        number_of_examples[partition_id] = len(labels)

    divergences_kl = compute_divergence_kl_for_clients(
        client_labels, dataset_config.num_classes
    )

    with round_metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.writer(metrics_file).writerow(
            ["server_round", "loss", "accuracy", "total_bytes_comunication"]
        )

    # Carregamento do modelo global
    global_model = Net(dataset)
    arrays = ArrayRecord(global_model.state_dict())

    sim_env = SimulationEnvironment(
        num_clients=num_partitions,
        divergences_kl=divergences_kl,
        number_of_examples=number_of_examples,
        costs_communication=generate_client_communication_costs(
            global_model, num_partitions
        ),
        max_rounds=num_rounds,
    )

    sim_env.reset()
    round_client_losses: dict[int, float] = {}

    def aggregate_training_metrics(
        records: list[RecordDict], weighting_key: str
    ) -> MetricRecord:
        """
        Agregação das métricas recebidas no treinamento dos clientes
        selecionados.

        Parâmetros:
            - records: lista de mensagens com métricas recebidas dos clientes
            - weighting_key: chave da métrica que será usada para agregar losses.

        Retorna: 
            - métricas recebidas agregadas
        """
        round_client_losses.clear()
        weighted_loss = 0.0
        total_examples = 0

        for record in records:
            metrics = record["metrics"]
            partition_id = int(metrics["partition-id"])
            train_loss = float(metrics["train_loss"])
            num_examples = int(metrics[weighting_key])
            
            round_client_losses[partition_id] = train_loss
            weighted_loss += train_loss * num_examples
            total_examples += num_examples

        aggregated: dict[str, int | float] = {weighting_key: total_examples}

        if total_examples:
            aggregated["train_loss"] = weighted_loss / total_examples

        return MetricRecord(aggregated)

    
    def evaluate_global_model(server_round: int, model_arrays: ArrayRecord):
        """
        Validação do modelo global.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - model_arrays: peso e parâmetros do modelo

        Retorna:
            - métricas da validação do modelo, com acurácia, loss e melhor
            acurácia global obtida até então
        """
        metrics = global_evaluate(
            server_round,
            model_arrays,
            sim_env,
            round_client_losses,
            dataset,
            num_partitions,
            alpha,
        )

        return metrics

    # Inicializa estratégia customizada
    strategy = SimulationFedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
        train_metrics_aggr_fn=aggregate_training_metrics,
        simulation_environment=sim_env,
    )

    # Inicia treinamento com estratégia
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_global_model,
    )

    if context.run_config["save-model"]:
        # Salva modelo final em disco
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

    # Resultado final do treinamento
    best_global_accuracy, final_global_loss, total_bytes = sim_env.get_final_metrics()
    megabytes = total_bytes / (1024**2)

    print("\nResumo do treinamento:")
    print(
        f"Melhor acurácia global: {best_global_accuracy:.4f} "
        f"({best_global_accuracy * 100:.2f}%)"
    )
    print(f"Loss global final: {final_global_loss:.6f}")
    print(
        "Dados contabilizados nos registros Flower "
        "(sem overhead do transporte): "
        f"{megabytes:.2f} MB"
    )

    generate_plots()


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    sim_env: SimulationEnvironment,
    round_client_losses: dict[int, float],
    dataset: str = DEFAULT_DATASET,
    num_partitions: int = 100,
    alpha: float = 0.5,
) -> MetricRecord:
    """
    Avalia modelo centralizado com todo o dataset de validação.

    Parâmetros:
        - server_round: número da rodada de treinamento
        - arrays: peso e parâmetros do modelo
        - sim_env: ambiente do gymnasium federado do treinamento
        - round_client_losses: lista de losses dos clientes na rodada

    Retorna:
        - métricas da validação: acurácia, loss e melhor loss até então
    """

    # Carrega o modelo e inicaliza com os pesos recebidos
    model = Net(dataset)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Carregar os dados do modelo
    test_dataloader = load_centralized_dataset(
        dataset, num_partitions, alpha
    )

    # Avaliação global do modelo
    test_loss, test_acc = test(model, test_dataloader, device)
    comm_bytes = sim_env.get_total_communication_bytes()

    if server_round > 0:
        with round_metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                [server_round, test_loss, test_acc, comm_bytes]
            )

        # Construção da máscara de clientes selecionados e losses
        clients_selected = np.zeros(sim_env.num_clients, dtype=bool)
        losses = np.zeros(sim_env.num_clients, dtype=np.float32)

        for partition_id, train_loss in round_client_losses.items():
            clients_selected[partition_id] = True
            losses[partition_id] = train_loss

        # Próximo passo no ambiente de simulação
        next_obs, reward, done, truncated, info = sim_env.step(
            loss=test_loss,
            accuracy=test_acc,
            clients_selected=clients_selected,
            client_losses=losses,
        )
        round_client_losses.clear()

        print(f"Recompensa da rodada {server_round}: {reward:.4f}")
        print(info)

    best_global_accuracy, _, _ = sim_env.get_final_metrics()

    # Retorna métricas de validação
    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
            "best_global_accuracy": best_global_accuracy,
        }
    )