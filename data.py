import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, cwd=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = os.path.join(cwd, "data")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, train=True, download=True)
        datasets.CIFAR10(root=self.data_path, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.cifar10_train = datasets.CIFAR10(root=self.data_path, train=True, transform=self.transform)
            self.cifar10_val = datasets.CIFAR10(root=self.data_path, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=1, shuffle=True)
    

