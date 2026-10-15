"""pytorchexample: A Flower / PyTorch app."""

import csv
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp

from pytorchexample.state_builder import (
    SimulationEnvironment,
)

from pytorchexample.task import (
    Net,
    load_centralized_dataset,
    partition_cache_dir,
    prepare_partitions,
    test,
    set_seed,
    compute_divergence_kl_for_clients,
    generate_client_communication_costs,
    SimulationFedAvg,
)

set_seed(42)  # Define a semente para reprodutibilidade

# Create ServerApp
app = ServerApp()
round_metrics_path = Path.cwd() / "round_metrics.csv"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    num_partitions = len(list(grid.get_node_ids()))
    alpha: float = context.run_config["alpha-dirichlet"]

    prepare_partitions(num_partitions, alpha)

    client_labels = []
    number_of_examples = np.zeros(num_partitions, dtype=np.int64)
    for partition_id in range(num_partitions):
        partition = load_from_disk(
            str(partition_cache_dir(num_partitions, alpha) / f"partition-{partition_id}")
        )
        labels = np.asarray(partition["train"]["label"], dtype=np.int64)
        client_labels.append(labels)
        number_of_examples[partition_id] = len(labels)

    global_labels = np.concatenate(client_labels)
    divergences_kl = compute_divergence_kl_for_clients(client_labels, global_labels)

    with round_metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.writer(metrics_file).writerow(
            ["server_round", "loss", "accuracy"]
        )

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    simulation_env = SimulationEnvironment(
        num_clients=num_partitions,
        divergences_kl=divergences_kl,
        number_of_examples=number_of_examples,
        costs_communication=generate_client_communication_costs(
            global_model, num_partitions
        ),
        max_rounds=num_rounds,
    )

    simulation_env.reset()
    round_client_losses: dict[int, float] = {}

    def aggregate_training_metrics(
        records: list[RecordDict], weighting_key: str
    ) -> MetricRecord:
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
        metrics = global_evaluate(
            server_round, model_arrays, simulation_env, round_client_losses
        )

        return metrics

    # Initialize FedAvg strategy
    strategy = SimulationFedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
        train_metrics_aggr_fn=aggregate_training_metrics,
        simulation_environment=simulation_env,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_global_model,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

    best_global_accuracy, final_global_loss, total_bytes = simulation_env.get_final_metrics()
    megabytes = total_bytes / (1024**2)


    print("\nResumo do treinamento:")
    print(
        f"Melhor acurácia global: {best_global_accuracy:.4f} "
        f"({best_global_accuracy * 100:.2f}%))"
    )
    print(f"Loss global final: {final_global_loss:.6f}")
    print(
        "Dados contabilizados nos registros Flower "
        "(sem overhead do transporte): "
        f"{megabytes:.2f} MB"
    )


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    simulation_env: SimulationEnvironment,
    round_client_losses: dict[int, float],
) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    if server_round > 0:
        with round_metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                [server_round, test_loss, test_acc]
            )

        clients_selected = np.zeros(simulation_env.num_clients, dtype=bool)
        losses = np.zeros(simulation_env.num_clients, dtype=np.float32)

        for partition_id, train_loss in round_client_losses.items():
            clients_selected[partition_id] = True
            losses[partition_id] = train_loss

        next_obs, reward, done, truncated, info = simulation_env.step(
            last_loss=test_loss,
            accuracy=test_acc,
            clients_selected=clients_selected,
            client_losses=losses,
        )
        round_client_losses.clear()

        print(f"Recompensa da rodada {server_round}: {reward:.4f}")
        print(info)

    best_global_accuracy, _, _ = simulation_env.get_final_metrics()
    # Return the evaluation metrics
    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
            "best_global_accuracy": best_global_accuracy,
        }
    )