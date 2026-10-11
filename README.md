---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the
`pytorchexample` package into the same Python environment used by `flwr run`.

```bash
python -m pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .  --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"  --stream
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Persisted partitions and round metrics

The first run saves each client's CIFAR-10 train/validation partition under
`data/federated_partitions/`. Later runs with the same number of clients and
Dirichlet alpha load those saved partitions instead of repartitioning the data.
The server writes global loss and accuracy for each completed round to
`round_metrics.csv`. The hyperparameter search writes its per-round metrics,
including the learning rate and proximal coefficient, to
`grid_search_round_metrics.csv`.

### Running without Ray metrics exporter warnings

If Ray's metrics exporter cannot connect in your environment and you do not need
Ray metrics, disable their collection for that run. Explicitly use Ray's current
behavior for zero-GPU workers to avoid its future-behavior warning:

```bash
RAY_enable_metrics_collection=0 \
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  flwr run . --stream
```

Disabling Ray metrics also disables their export to monitoring tools. To remove
the Hugging Face Hub's anonymous-request warning, authenticate once in the
environment used to run Flower:

```bash
hf auth login
```

This stores the token in the Hugging Face user cache, not in the project.
Anonymous access to public datasets still works, but may be subject to lower
rate limits.

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
