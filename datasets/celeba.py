from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from datasets.sampler import ResumableSeedableSampler


def get_celeba_dataloader(batch_size, seed, data_dir="data/", normalize: bool = True):
    """
    Builds a dataloader with all images from the CelebA dataset.
    Args:
        data_dir: Directory where the data is stored.
        batch_size: Size of the batches

    Returns:
        DataLoader: DataLoader object containing the dataset.

    """

    if normalize:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.CenterCrop((178, 178)),
                transforms.Resize((64, 64)),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop((178, 178)),
                transforms.Resize((64, 64)),
            ]
        )

    path = Path(data_dir)

    dataset = CelebA(root=path, split="all", download=True, transform=data_transforms)

    sampler = ResumableSeedableSampler(dataset, seed=seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=sampler,
    )
