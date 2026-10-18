import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from structure.classes.neural_network import Net
from structure.utils.dataset_config import DEFAULT_DATASET
from structure.utils.data import load_data
from structure.utils.training import test as test_fn
from structure.utils.training import train as train_fn
from structure.utils.common import set_seed

set_seed(42)  # Define a semente para reprodutibilidade

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """
    Treinamento do modelo com dados locais.

    Parâmetros:
        - msg: mensagem enviada pelo servidor, contendo máscara de
        clientes elegíveis
        - context: contexto do treinamento, com o número do cliente e a
        quantidade de partições

    Retorna:
        - mensagem com o modelo treinado e métricas obtidas durante o
        treinamento, como se o cliente é elegível, a loss do treinamento,
        número de exemplos do dataset local e número do cliente
    """

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset = context.run_config.get("dataset", DEFAULT_DATASET)
    eligible_clients = msg.content["config"]["eligible-clients-mask"]
   
   # Caso não esteja disponível, retorna sem treinar
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

    # Carrega o modelo e o atualiza com os pesos recebidos
    model = Net(dataset)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Carrega dados do particionamento
    batch_size = context.run_config["batch-size"]
    alpha_dirichlet = context.run_config["alpha-dirichlet"]
    num_local_epochs = context.run_config["local-epochs"]

    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        alpha_dirichlet,
        dataset=dataset,
    )

    # Chama função de treinamento
    train_loss, _ = train_fn(
        model,
        trainloader,
        num_local_epochs,
        msg.content["config"]["lr"],
        device,
    )

    # Construção da mensagem e retorno para o servidor
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


""" @app.evaluate()
def evaluate(msg: Message, context: Context):
    Validação do modelo global em dados locais.

    Parâmetros:
        - msg: mensagem enviada pelo servidor
        - context: contexto do treinamento, com o número do cliente, a
        quantidade de partições, tamanho do batch e alpha do dirichlet

    Retorna:
        - mensagem com as métricas obtidas durante o treinamento,
        como a loss de validação e número de exemplos do dataset local

    # Carrega o modelo e o atualiza com os pesos recebidos
    dataset = context.run_config.get("dataset", DEFAULT_DATASET)
    model = Net(dataset)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Carrega dados de validação do cliente
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha_dirichlet = context.run_config["alpha-dirichlet"]
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        alpha_dirichlet,
        dataset=dataset,
    )

    # Chamada de validação do modelo sobre os dados do cliente
    eval_loss, _ = test_fn(
        model,
        valloader,
        device,
    )

    # Construção e retorno da mensagem com métricas para o servidor
    metrics = {
        "eval_loss": eval_loss,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg) """