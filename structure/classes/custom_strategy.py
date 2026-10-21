from flwr.app import (ArrayRecord, ConfigRecord, Message, MetricRecord)
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.serverapp import Grid
from collections.abc import Iterable

from structure.classes.federated_classes import SimulationEnvironment
from ..utils.common import _count_message_record_bytes


class SimulationFedAvg(FedAvg):
    """ FedAvg customizado para disponibilidade de clientes. """

    def __init__(self, *args, simulation_environment: SimulationEnvironment, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_environment = simulation_environment

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Configuração de treinamento: envia clientes disponíveis nas mensagens de
        treinamento e grava bytes de comunicação das mensagens enviadas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - arrays: pesos e parâmetros do modelo
            - config: hiperparâmetros e configurações do treinamento
            - grid: infraestrutura da rede dos nós do sistema Flower

        Retorna:
            - lista de mensagens a serem enviadas aos clientes
        """

        config["eligible-clients-mask"] = (
            self.simulation_environment.eligible_clients_mask.astype(int).tolist()
        )
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(messages)
        )
        return messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Agregação do treinamento: recebe as respostas dos clientes, aumenta o 
        número de bytes de comunicação e envia apenas as respostas válidas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - replies: mensagens retornadas pelos clientes

        Retorna:
            - lista de mensagens retornadas pelos clientes disponíveis
        """
        replies = list(replies)
        eligible_replies = [
            reply
            for reply in replies
            if not reply.has_error() and bool(reply.content["metrics"]["eligible"])
        ]
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(eligible_replies)
        )
        
        return super().aggregate_train(server_round, eligible_replies)


class SimulationFedProx(SimulationFedAvg):
    """ FedProx customizado para disponibilidade de clientes. """

    def __init__(
        self,
        *args,
        proximal_mu: float = 0.1,
        simulation_environment: SimulationEnvironment,
        **kwargs,
    ):
        
        super().__init__(*args, simulation_environment=simulation_environment, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Configuração de treinamento: envia clientes disponíveis nas mensagens de
        treinamento e o proximal_mu, e grava bytes de comunicação das mensagens enviadas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - arrays: pesos e parâmetros do modelo
            - config: hiperparâmetros e configurações do treinamento
            - grid: infraestrutura da rede dos nós do sistema Flower

        Retorna:
            - lista de mensagens a serem enviadas aos clientes
        """
        config["proximal-mu"] = self.proximal_mu

        return super().configure_train(server_round, arrays, config, grid)


class SimulationPowerOfChoice(FedAvg):
    """ Power-of-Choice (POC) customizado para disponibilidade de clientes. """

    def __init__(
        self,
        *args,
        num_selected: int = 10,
        simulation_environment: SimulationEnvironment,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.simulation_environment = simulation_environment
        self.num_selected = num_selected  
        self.latest_client_losses: dict[int, float] = {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Configuração de treinamento: envia clientes disponíveis nas mensagens de
        treinamento, seleciona os clientes para mandar mensagens selecionando os
        top-K com maiores losses dentre d disponíveis e grava bytes de comunicação
        das mensagens enviadas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - arrays: pesos e parâmetros do modelo
            - config: hiperparâmetros e configurações do treinamento
            - grid: infraestrutura da rede dos nós do sistema Flower

        Retorna:
            - lista de mensagens a serem enviadas aos clientes
        """

        """
        Gera as mensagens candidatas, ordena pela maior loss registrada
        e seleciona apenas os K clientes prioritários.
        """
        config["eligible-clients-mask"] = (
            self.simulation_environment.eligible_clients_mask.astype(int).tolist()
        )

        # Gera as mensagens candidatas da estratégia base (grupo de tamanho d)
        candidate_messages = list(super().configure_train(server_round, arrays, config, grid))

        # Ordena as mensagens pela maior loss (clientes sem histórico recebem inf para exploração inicial)
        sorted_messages = sorted(
            candidate_messages,
            key=lambda msg: self.latest_client_losses.get(
                msg.metadata.dst_node_id, float("inf")
            ),
            reverse=True,
        )

        # Filtra apenas as K mensagens dos clientes escolhidos
        selected_messages = sorted_messages[: self.num_selected]

        # Registra os bytes transmitidos APENAS para as K mensagens efetivamente enviadas
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(selected_messages)
        )

        return selected_messages

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Agrega o treinamento, atualiza o histórico de loss dos clientes
        e contabiliza a comunicação das respostas válidas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - replies: mensagens retornadas pelos clientes

        Retorna:
            - lista de mensagens retornadas pelos clientes disponíveis
        """
        replies = list(replies)
        eligible_replies = [
            reply
            for reply in replies
            if not reply.has_error() and bool(reply.content["metrics"]["eligible"])
        ]

        # Atualiza a tabela de losses dos clientes que concluíram o treino
        for reply in eligible_replies:
            node_id = reply.metadata.src_node_id
            metrics = reply.content["metrics"]
            if "train_loss" in metrics:
                self.latest_client_losses[node_id] = float(metrics["train_loss"])

        # Registra bytes das mensagens de retorno recebidas
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(eligible_replies)
        )

        return super().aggregate_train(server_round, eligible_replies)