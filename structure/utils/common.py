import numpy as np
import random
import torch
from flwr.app import Message
from collections.abc import Iterable

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _count_message_record_bytes(messages: Iterable[Message]) -> int:
    """Count Flower record payloads, excluding message and transport overhead."""
    return sum(
        record.count_bytes()
        for message in messages
        if not message.has_error()
        for record in message.content.values()
    )