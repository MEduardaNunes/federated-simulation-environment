from flwr.app import (ArrayRecord, ConfigRecord, Message, MetricRecord)
from flwr.serverapp.strategy import FedAvg
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
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(replies)
        )
        eligible_replies = [
            reply
            for reply in replies
            if not reply.has_error() and bool(reply.content["metrics"]["eligible"])
        ]
        return super().aggregate_train(server_round, eligible_replies)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """
        Configuração de validação: envia clientes disponíveis nas mensagens de
        validação e grava bytes de comunicação das mensagens enviadas.

        Parâmetros:
            - server_round: número da rodada de validação
            - arrays: pesos e parâmetros do modelo
            - config: hiperparâmetros e configurações da validação
            - grid: infraestrutura da rede dos nós do sistema Flower

        Retorna:
            - lista de mensagens a serem enviadas aos clientes
        """
        messages = list(
            super().configure_evaluate(server_round, arrays, config, grid)
        )
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(messages)
        )
        return messages

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """
        Agregação da validação: recebe as respostas dos clientes, aumenta o 
        número de bytes de comunicação e envia as respostas.

        Parâmetros:
            - server_round: número da rodada de treinamento
            - replies: mensagens retornadas pelos clientes

        Retorna:
            - lista de mensagens retornadas pelos clientes 
        """

        replies = list(replies)
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(replies)
        )
        return super().aggregate_evaluate(server_round, replies)