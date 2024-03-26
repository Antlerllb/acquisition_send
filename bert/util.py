import pandas as pd
import os
import random
import numpy as np
import torch
from typing import Literal, List, Union, Optional
from torch import nn


# 冻结或更新模型的参数
def update_params(model: nn.Module, frozen_names: Optional[list] = None, loaded_params: Optional[dict] = None):
    for name, param in model.named_parameters():
        # 更新参数
        if loaded_params is not None and name in loaded_params:
            param.data.copy_(loaded_params[name])

        # 冻结参数
        if frozen_names is not None and any(frozen_name in name for frozen_name in frozen_names):
            param.requires_grad = False


def init_seed(seed: int, framework: Literal['pt', 'tf'] = 'pt') -> None:
    """
    批量初始化深度学习框架的种子值。

    Args:
        seed (int): 种子值。
        framework (Literal['pt', 'tf'], default to 'pt'): PyTorch/TensorFlow框架。

    Returns:
        None

    Examples:
        >>> init_seed(42, 'tf')
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 哈希种子
    np.random.seed(seed)
    if framework == 'pt':
        torch.manual_seed(seed)  # CPUws
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # 多 GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif framework == 'tf':
        ...
