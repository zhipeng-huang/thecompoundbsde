import torch
from torch import optim


def build_optimizer(loaded_model, optim_name, ini_lr):
    """
    Args:
        loaded_model: torch.nn.Module
        optim_name: str, e.g. "adam", "rmsprop", "sgd", "adamw", "rprop"
        ini_lr: float

    Returns:
        torch.optim.Optimizer
    """
    if optim_name is None:
        raise ValueError("optim_name must be provided (e.g. 'adam', 'rmsprop').")

    if not isinstance(optim_name, str):
        raise TypeError(f"optim_name must be a string, got {type(optim_name)}.")

    name = optim_name.lower()
    if name == "adam":
        print("Using Adam optimizer")
        return optim.Adam(
            loaded_model.parameters(),
            lr=ini_lr,
        )

    if name == "adamw":
        print("Using AdamW optimizer")
        return optim.AdamW(
            loaded_model.parameters(),
            lr=ini_lr,
            weight_decay=1e-2,
        )

    if name == "sgd":
        print("Using sgd optimizer")
        return optim.SGD(
            loaded_model.parameters(),
            lr=ini_lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)

    if name == "rmsprop":
        print("Using rmsprop optimizer")
        return optim.RMSprop(
            loaded_model.parameters(),
            lr=ini_lr,
            alpha=0.99,
            eps=1e-8,
            momentum=0.0,
            weight_decay=0.0,
            centered=False,
        )

    if name == "rprop":
        print("Using rprop optimizer")
        return optim.Rprop(
            loaded_model.parameters(),
            lr=ini_lr,             # initial step size
            etas=(0.5, 1.2),       # decrease / increase factors
            step_sizes=(1e-6, 50), # min / max step size
        )

    raise ValueError(f"Unknown optimizer name: {optim_name}")
