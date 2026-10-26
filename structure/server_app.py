import csv
import os
import ast
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from datasets import load_from_disk
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from structure.classes.federated_classes import SimulationEnvironment
from structure.classes.custom_strategy import *
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
from structure.utils.final_savings import (
    aggregate_metrics_and_plots,
    save_config_metrics_and_plots,
)

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"]="0"


# Create ServerApp
app = ServerApp()
round_metrics_path = Path.cwd() / "round_metrics.csv"


@app.main()
def main(grid: Grid, context: Context) -> None:
    seeds = ast.literal_eval(
        context.run_config.get("seeds", "[42]")
    )
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 or seed > 2**32 - 1 for seed in seeds)
    ):
        raise ValueError(
            "seeds must be a non-empty list of integers from 0 to 2**32 - 1"
        )

    global round_metrics_path
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = Path("results") / run_timestamp

    for seed in seeds:
        seed_output_dir = run_output_dir / f"seed-{seed}"
        metrics_dir = seed_output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        round_metrics_path = Path.cwd() / f"round_metrics_seed-{seed}.csv"
        with round_metrics_path.open(
            "w", newline="", encoding="utf-8"
        ) as metrics_file:
            csv.writer(metrics_file).writerow(
                ["server_round", "loss", "accuracy", "total_bytes_comunication"]
            )
        run_training_for_seed(grid, context, seed, seed_output_dir)

    if context.run_config["save-result"]:
        aggregate_metrics_and_plots(
            input_dir=run_output_dir,
            output_dir=run_output_dir / "aggregate",
        )


def run_training_for_seed(
    grid: Grid, context: Context, seed: int, seed_output_dir: Path
) -> None:
    """
    Coordena um treinamento federado para uma seed específica.

    Parâmetros:
        - grid: infraestrutura da rede dos nós do sistema Flower
        - context: contexto da simulação, contendo principais hiperparâmetros
        para o treinamento
        - seed: semente do experimento
        - seed_output_dir: diretório dos resultados desta seed
    """
    set_seed(seed)

    # Criação das partições em disco e leitura das labels dos dados de cada
    # cliente para cálculo de divergências KL
    num_partitions = len(list(grid.get_node_ids()))
    dataset: str = context.run_config.get("dataset", "CIFAR-10")
    type_partition: str = context.run_config.get("type-partition", "IID")
    alpha: float | None = None

    if type_partition == "NO_IID":
        alpha = context.run_config["alpha-dirichlet"]

    dataset = dataset + "_" + type_partition
    dataset_config = get_dataset_config(dataset)

    prepare_partitions(num_partitions, alpha, seed=seed, dataset=dataset)
    client_labels = []
    number_of_examples = np.zeros(num_partitions, dtype=np.int64)
    for partition_id in range(num_partitions):
        partition = load_from_disk(
            str(
                partition_cache_dir(
                    num_partitions, alpha, seed=seed, dataset=dataset
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

    # Carregamento do modelo global
    num_rounds: int = context.run_config["num-server-rounds"]

    global_model = Net(dataset)
    arrays = ArrayRecord(global_model.state_dict())

    sim_env = SimulationEnvironment(
        num_clients=num_partitions,
        divergences_kl=divergences_kl,
        number_of_examples=number_of_examples,
        costs_communication=generate_client_communication_costs(
            global_model, num_partitions, seed=seed
        ),
        max_rounds=num_rounds,
        seed=seed,
    )

    sim_env.reset(seed=seed)
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
            seed,
        )

        return metrics

    # Inicializa estratégia customizada
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    strategy_name = context.run_config["strategy"]

    if strategy_name == "FedAvg":
        strategy = SimulationFedAvg(
            fraction_evaluate=fraction_evaluate,
            fraction_train=fraction_train,
            train_metrics_aggr_fn=aggregate_training_metrics,
            simulation_environment=sim_env,
        )

    elif strategy_name == "FedProx":
        proximal_mu = context.run_config["proximal-mu"]
        
        strategy = SimulationFedProx(
            fraction_evaluate=fraction_evaluate,
            fraction_train=fraction_train,
            train_metrics_aggr_fn=aggregate_training_metrics,
            simulation_environment=sim_env,
            proximal_mu=proximal_mu
        )

    elif strategy_name == "POC":
        K : int = context.run_config["K"]
        
        strategy = SimulationPowerOfChoice(
            fraction_evaluate=fraction_evaluate,
            fraction_train=fraction_train,
            train_metrics_aggr_fn=aggregate_training_metrics,
            num_selected=K,
            simulation_environment=sim_env,
        )

    else:
        print("Estratégia não conhecida")
        return 0

    # Inicia treinamento com estratégia
    lr: float = context.run_config["learning-rate"]

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "seed": seed}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_global_model,
    )

    if context.run_config["save-model"]:
        # Salva modelo final em disco
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, seed_output_dir / f"final_model_seed-{seed}.pt")

    # Resultado final do treinamento
    best_global_accuracy, final_global_loss, total_bytes = sim_env.get_final_metrics()
    megabytes = total_bytes / (1024**2)

    print(f"\nResumo do treinamento:\n"
          f"Melhor acurácia global: {best_global_accuracy:.4f}"
          f"({best_global_accuracy * 100:.2f}%)\n"
          f"Loss global final: {final_global_loss:.6f}\n"
          "Total de bytes de comunicação "
          f"{megabytes:.2f} MB"
    )

    # Escrita dos resultados finais em disco
    metrics_path = seed_output_dir / "metrics" / round_metrics_path.name
    shutil.copy2(round_metrics_path, metrics_path)
    round_metrics_path.unlink()

    if context.run_config["save-result"]:
        save_config_metrics_and_plots(
            csv_path=str(metrics_path),
            output_dir=str(seed_output_dir),
        )


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    sim_env: SimulationEnvironment,
    round_client_losses: dict[int, float],
    dataset: str = DEFAULT_DATASET,
    num_partitions: int = 100,
    alpha: float = 0.5,
    seed: int = 42,
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
        dataset, num_partitions, alpha, seed
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