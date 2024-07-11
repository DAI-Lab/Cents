from typing import Tuple

import torch


class ACGAN(object):

    def __init__(
        self,
        input_dim: Tuple[torch.int64, torch.int64],
        window_length: torch.int64,
        weight_path: torch.tensor,
        code_size=64,
        learning_rate=1e-4,
        batch_size=32,
    ) -> None:
        pass
