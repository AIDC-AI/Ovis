import os
from importlib import import_module
import torch


def rank0_print(*args):
    if int(os.getenv("LOCAL_PROCESS_RANK", os.getenv("LOCAL_RANK", 0))) == 0:
        print(*args)


def rankN_print(*args):
    rank = int(os.getenv("LOCAL_PROCESS_RANK", os.getenv("LOCAL_RANK", 0)))
    print(f'<R{rank}>', *args)


def smart_unit(num):
    if num / 1.0e9 >= 1:
        return f'{num / 1.0e9:.2f}B'
    else:
        return f'{num / 1.0e6:.2f}M'



def replace_torch_load_with_weights_only_false():
    original_torch_load = torch.load

    def torch_load_with_weights_only_false(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    # 替换 torch.load
    torch.load = torch_load_with_weights_only_false
