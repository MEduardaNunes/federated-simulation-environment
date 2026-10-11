"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

from pytorchexample.task import set_seed

set_seed(42)  # Define a semente para reprodutibilidade

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    eligible_clients = msg.content["config"]["eligible-clients-mask"]
   
    if not eligible_clients[partition_id]:
        metrics = MetricRecord(
            {
                "eligible": 0,
                "train_loss": 0.0,
                "num-examples": 0,
                "partition-id": partition_id,
            }
        )
        return Message(
            content=RecordDict({"arrays": msg.content["arrays"], "metrics": metrics}),
            reply_to=msg,
        )

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    batch_size = context.run_config["batch-size"]
    alpha_dirichlet = context.run_config["alpha-dirichlet"]
    num_local_epochs = context.run_config["local-epochs"]

    trainloader, _ = load_data(partition_id, num_partitions, batch_size, alpha_dirichlet)

    # Call the training function
    train_loss, _ = train_fn(
        model,
        trainloader,
        num_local_epochs,
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "eligible": 1,
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": partition_id,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha_dirichlet = context.run_config["alpha-dirichlet"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, alpha_dirichlet)

    # Call the evaluation function
    eval_loss, _ = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)