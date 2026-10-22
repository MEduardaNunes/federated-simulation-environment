---
dataset: [CIFAR-10, FEMNIST]
framework: [torch, torchvision, flower, gymnasium]
---

# Aprendizado Federado com algoritmo customizado de seleção de clientes com agente baseado em Reinforcement Learning

Esse repositório busca implementar um ambiente federado que permita o treinamento em clientes de uma rede neural para classificar imagens de um dataset (suporte para CIFAR-10 e FEMNIST). Além disso, também é possível fazer partições IID e não-IID dos datasets entre os clientes, que será explicado posteriormente. Os algoritmos de seleção de clientes possíveis são: FedAvg, FedProx e Power-of-Choic (posteriormente terá o agente baseado em Aprendizado por Reforço).


## Set up do projeto
### Criação e ativação do ambiente virtual

Recomenda-se criar um ambiente virtual para rodar o experimento em uma pasta acima do projeto. Para isso, rode o comando:

```bash
python3 -m venv .venv       # Cria o ambiente virtual
source .venv/bin/activate   # Ativa o ambiente virtual
```

No final desse passo, devemos ter uma organização do diretório da seguinte forma:

```shell
diretorio-pai
├── mes-05
└── .venv
```

### Instalando dependência

Depois de ativar o ambiente virtual, instale as dependências definidas em `pyproject.toml`.

```bash
pip install -e .
```


## Definindo os hiperparâmetros, datasets e tipo de particionamento da simulação

No arquivo `pyproject.toml`, na seção `[tool.flwr.app.config]`, são definidas as variáveis do projeto que irão definir os valores dos hiperparâmetros, qual dataset será utilizado, bem como o tipo do seu particionamento. À seguir, são explicados todos os parâmetros e seus significados.

### num-server-rounds
Define o número de rodadas de treinamento do servidor. Seu valor deve ser um número inteiro maior que zero.

### fraction-evaluate
Define a fração de clientes selecionados para realizar a avaliação após cada rodada de treinamento. Seu valor deve ser um número decimal no intervalo [0, 1].

### fraction-train
Define a fração de clientes selecionados para realizar o treinamento local em uma rodada. Seu valor deve ser um número decimal no intervalo [0, 1]. Caso a estratégia utilizada seja PowerOfChoice, deve ter o valor igual a d/C.

### local-epochs
Define o número de épocas locais de treinamento realizadas por um cliente. Seu valor deve ser um número inteiro maior que zero.

### learning-rate
Define o valor da taxa de aprendizado para o treinamento supervisionado. Seu valor deve ser um número decimal positivo, idealmente no intervalo [0, 1].

### batch-size
Define a quantidade de exemplos a serem selecionados em cada época local em um cliente. Seu valor deve ser um número inteiro maior que zero.

### dataset
Define o dataset a ser utilizado. Pode assumir apenas um valor dentro os dois disponíveis: 
`"CIFAR-10"` e `"FEMNIST"`.

### type-partition
Define o tipo de particionamento a ser utilizado. Pode assumir apenas um valor dentro dos dois disponíveis: `"IID"` e `"NO_IID"`.

strategy = "POC"
proximal-mu = 0.75

### alpha-dirichlet
Define o valor do alpha para o termo Dirichlet, utilizado para controlar a diversidade entre os dados dos clientes. Deve ser definido apenas se o particionamento for não-IID. Seu valor deve ser um número decimal positivo no intervalo [0, 1].

### strategy
Define a estratégia a ser utilizada para a agregação das atualizações recebidas pelo clientes. Pode assumir apenas um valor dentro os disponíveis, explicados à seguir:
- `"FedAvg"`: utiliza FedAvg,
- `"FedProx"`: utiliza FedProx (deve-se definir `"proximal-mu"`),
- `"POC"`: utiliza PowerOfChoice (deve-se definir `"K"`).

### proximal-mu
Define o valor do hiperparâmetro que controle a força do termo proximal na função de perda. Seu valor deve ser um número decimal maior que zero, idealmente no intervao [0, 1].

### K
Define o valor de clientes a serem selecionados efetivamente dentre os D disponíveis na estratégia PowerOfChoice. Seu valor deve ser um número inteiro no intervalo [1, D].

### save-model
Define se o modelo final deve ser salvo em disco ou não. Seu valor deve ser `true` ou `false`.

### save-result
Define se os plots, o arquivo de métricas de cada rodada (com loss, acurácia e total de bytes comunicados) e a configuração do projeto devem ser salvos ou não em disco. Seu valor deve ser `true` ou `false`.


## Exemplos de configurações base para cada estratégia

À seguir, será apresentada a seção `[tool.flwr.app.config]` do arquivo `pyproject.toml` com os hiperparâmetros que devem ser definidos para cada estratégia possível (FedAvg, FedProx e POC).

### Configuração base FedAvg
```shell
[tool.flwr.app.config]
num-server-rounds = 100 
fraction-evaluate = 0.0
fraction-train = 0.1
local-epochs = 1
learning-rate = 0.05
batch-size = 64

dataset = "CIFAR-10"
type-partition = "NO_IID"
strategy = "FedAvg"
alpha-dirichlet = 0.1

save-model = false
save-result = true
```

### Configuração base FedProx
```shell
[tool.flwr.app.config]
num-server-rounds = 100 
fraction-evaluate = 0.0
fraction-train = 0.1
local-epochs = 1
learning-rate = 0.05
batch-size = 64

dataset = "CIFAR-10"
type-partition = "NO_IID"
strategy = "FedProx"
alpha-dirichlet = 0.1
proximal-mu = 0.01

save-model = false
save-result = true
```

### Configuração base PowerOfChoice (POC)
```shell
[tool.flwr.app.config]
num-server-rounds = 100 
fraction-evaluate = 0.0
fraction-train = 0.75
local-epochs = 1
learning-rate = 0.05
batch-size = 64

dataset = "CIFAR-10"
type-partition = "NO_IID"
strategy = "POC"
alpha-dirichlet = 0.1
K = 10

save-model = false
save-result = true
```

## Rodando o projeto

Para rodar a simulação e visualizar os logs do treinamento, rode o comando:

```bash
flwr run .  --stream
```

Você pode sobrescrever hiperparâmetros definidos na seção `[tool.flwr.app.config]` do arquivo `pyproject.toml` com o comando:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"  --stream
```