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

### seeds
Define uma lista não vazia de seeds inteiras, entre `0` e `2**32 - 1`. O projeto executa um treinamento completo para cada seed, usando-a na inicialização do modelo, no particionamento e carregamento dos dados e nos componentes aleatórios da simulação. Os arquivos de métricas e modelos são identificados pela seed para evitar sobrescrita. Exemplo:

```toml
seeds = "[42, 123, 456]"
```

Os resultados de uma execução são salvos em `results/<horario-inicial>/seed-<valor-da-seed>`, mantendo os arquivos de cada seed organizados dentro da pasta identificada pelo horário de início do treinamento. Os arquivos `round_metrics` ficam em `metrics/` dentro de cada diretório de seed.

Ao executar o treinamento normal, as métricas por rodada entre as seeds são consolidadas automaticamente após a conclusão de todas elas. Os arquivos `aggregated_metrics.csv`, `accuracy_curve.png`, `loss_curve.png` e `communication_bytes.png` são salvos em `results/<horario-inicial>/aggregate/`. Os gráficos mostram a média e a faixa de um desvio padrão entre os CSVs, com resolução de 300 DPI.

Para executar somente a agregação de uma execução já concluída, sem iniciar o treinamento:

```bash
python -m structure.utils.final_savings \
  --input-dir results/<horario-inicial>
```

O comando lê `seed-*/metrics/round_metrics*.csv` e grava os arquivos agregados em `results/<horario-inicial>/aggregate/`. Opcionalmente, informe `--output-dir <pasta>` para alterar a saída ou `--dpi <valor>` para mudar a resolução dos gráficos.

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

Se o cache de partições para a configuração atual (dataset, tipo, número de clientes, seed e, para NO_IID, alpha) não existir, o sistema o gera usando o dataset fonte. Ele procura primeiro uma cópia salva em `data/<nome-do-dataset>` — por exemplo, `data/cifar10` para CIFAR-10 — que deve ser um dataset salvo com `save_to_disk()` e conter uma divisão `train` se for um `DatasetDict`. Se essa cópia local não existir, o dataset é carregado do Hugging Face Hub e requer conexão com a internet.

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

### Executando no cluster com Slurm

O script `submit_flower.sh` submete a simulação com `sbatch`. Ele usa o ambiente
virtual externo recomendado acima (`../.venv`); para usar outro ambiente, defina
`FLOWER_VENV` ao submeter. Crie o ambiente e instale as dependências uma única vez
antes da submissão, seguindo as instruções de instalação deste README.

```bash
sinfo
scontrol show partition
sbatch submit_flower.sh
squeue -u "$USER"
JOBID=123456  # substitua pelo ID retornado por sbatch
tail -f "slurm-flower-${JOBID}.out"
tail -f "slurm-flower-${JOBID}.err"
scancel "$JOBID"
```

O script não escolhe partição nem solicita GPU. Ele reserva seis CPUs e 16 GiB
por até quatro horas, limita o Ray a quatro CPUs e quatro SuperNodes e atribui
duas CPUs a cada cliente virtual, executando clientes em CPU. Quatro SuperNodes
correspondem às partições existentes no cache de dados deste checkout; ajuste
essa configuração se o experimento exigir outra quantidade de clientes. Os
recursos são valores iniciais conservadores, não uma garantia de disponibilidade
ou desempenho do cluster. As saídas continuam em `results/` e os logs do Slurm
ficam no diretório de onde `sbatch` foi executado. O script impede execuções
simultâneas usando o mesmo checkout para proteger os arquivos temporários de
métricas. O estado do SuperLink local fica isolado por job em `results/`, e a
instalação automática de dependências do Flower é desativada: por isso o ambiente
virtual precisa conter as dependências do projeto antes da submissão. O Slurm
deve encerrar os processos do job ao final; verifique a política local para
processos auxiliares em background.