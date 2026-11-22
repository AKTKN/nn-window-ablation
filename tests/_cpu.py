"""PyTorch thread setup for processes running ordinary tests."""


def configure_torch_threads() -> None:
    import torch

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    # PyTorch only allows the inter-op setter to run once in a process.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
