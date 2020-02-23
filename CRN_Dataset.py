import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
from random import random
from matplotlib import pyplot as plt


class CRNDataset(Dataset):
    def __init__(
        self,
        max_input_height_width: tuple,
        root: str,
        split: str,
        should_flip: bool,
        subset_size: int,
        noise: bool,
    ):
        super(CRNDataset, self).__init__()
        self.num_input_classes: int = 34
        self.num_output_classes: int = 20

        self.should_flip: bool = should_flip
        self.subset_size: int = subset_size
        self.noise: bool = noise

        self.dataset: Cityscapes = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )

        self.max_input_height_width = max_input_height_width

        self.indices = torch.tensor(
            [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        )

        self.image_resize_transform = transforms.Compose(
            [
                transforms.Resize(max_input_height_width, Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    max_input_height_width,
                    Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).long()[0]),
                transforms.Lambda(lambda x: one_hot(x, self.num_input_classes)),
                transforms.Lambda(lambda x: torch.index_select(x, 2, self.indices)),
                transforms.Lambda(lambda x: x.transpose(0, 2).transpose(1, 2)),
                transforms.Lambda(lambda x: CRNDataset.__add_remaining_layer__(x)),
                transforms.Lambda(lambda x: x.float()),
            ]
        )

    def __getitem__(self, index):
        img, msk = self.dataset.__getitem__(index)

        flip: bool = random()

        if self.should_flip and flip > 0.5:
            img = transforms.functional.hflip(img)
            msk = transforms.functional.hflip(msk)

        img: torch.Tensor = self.image_resize_transform(img)
        msk: torch.Tensor = self.mask_resize_transform(msk)
        if self.noise and torch.rand(1).item() > 0.5:
            img = img + torch.normal(0, 0.02, img.size())
            img[img > 1] = 1
            img[img < -1] = -1
        if self.noise and torch.rand(1).item() > 0.5:
            mean_range: float = (torch.rand(1).item() * 0.2) + 0.7
            msk_noise = torch.normal(mean_range, 0.1, msk.size())
            msk_noise = msk_noise.int().float()
            # print(msk_noise.sum() / self.num_classes)
            msk = msk + msk_noise
        return img, msk

    def __len__(self):

        if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
            return self.dataset.__len__()
        else:
            return self.subset_size

    @staticmethod
    def __add_remaining_layer__(x: torch.Tensor):
        layer: torch.Tensor = torch.zeros_like(x[0])
        layer[x.sum(dim=0) == 0] = 1
        return torch.cat((x, layer.unsqueeze(dim=0)), dim=0)
