from torch.utils.data import DataLoader
import torch


class IterLoader(DataLoader):
    def __init__(self, steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = steps
        self._count = 0
        self._data_iter = super().__iter__()

    def __iter__(self):
        self._count = 0
        self._data_iter = super().__iter__()
        return self

    def __next__(self):
        self._count += 1
        if self._count > self.steps:
            raise StopIteration
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = super().__iter__()
            return next(self._data_iter)
    
    def __len__(self):
        return self.steps


def dict_collate(batch):
    """Collate function for DataLoader."""
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0].keys()}


def dict_tensor_to_device(
    data: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move dictionary to device."""
    return {k: v.to(device) for k, v in data.items()}
