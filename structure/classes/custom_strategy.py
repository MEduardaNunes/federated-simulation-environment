from flwr.app import (ArrayRecord, ConfigRecord, Message, MetricRecord)
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp import Grid
from collections.abc import Iterable

from structure.classes.federated_classes import SimulationEnvironment
from ..utils.common import _count_message_record_bytes


class SimulationFedAvg(FedAvg):
    """FedAvg that enforces the simulation environment's client availability."""

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
        replies = list(replies)
        self.simulation_environment.record_communication_bytes(
            _count_message_record_bytes(replies)
        )
        return super().aggregate_evaluate(server_round, replies)