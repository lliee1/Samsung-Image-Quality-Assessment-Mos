from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from MANIQA.data.koniq10k.koniq10k import (
    MyDataset,
    MyDataset_test,
    MyDataset_loss_check,
    MyDataset_with_blur,
    MyDataset_384,
    MyDataset_384_test,
    MyDataset_with_blur_384,
)
from MANIQA.utils.process import (
    RandCrop,
    Normalize,
    ToTensor,
    RandHorizontalFlip,
    ToTensor_test,
    Normalize_test,
)


class Maniqa_384DataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_csv_file: str = "",
        valid_csv_file: str = "",
        test_csv_file: str = "",
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transforms = transforms.Compose(
            [
                RandCrop(patch_size=384),
                Normalize(0.5, 0.5),
                RandHorizontalFlip(prob_aug=0.7),
                ToTensor(),
            ]
        )
        self.val_transforms = transforms.Compose(
            [RandCrop(patch_size=384), Normalize(0.5, 0.5), ToTensor()]
        )
        self.test_transforms = transforms.Compose(
            [Normalize_test(0.5, 0.5), ToTensor_test()]
        )

        # self.data_train: Optional[Dataset] = MyDataset(
        #     csv_file=train_csv_file, transform=self.train_transforms
        # )
        # self.data_val: Optional[Dataset] = MyDataset(
        #     csv_file=valid_csv_file, transform=self.val_transforms
        # )
        self.data_train: Optional[Dataset] = MyDataset_with_blur_384(
            csv_file=train_csv_file, transform=self.train_transforms
        )
        self.data_val: Optional[Dataset] = MyDataset_384(
            csv_file=valid_csv_file, transform=self.val_transforms
        )
        self.data_test: Optional[Dataset] = MyDataset_384_test(
            csv_file=test_csv_file, transform=self.test_transforms
        )
        # self.data_test: Optional[Dataset] = MyDataset_loss_check(
        #     csv_file=valid_csv_file, transform=self.val_transforms
        # )

    @property
    def num_classes(self) -> int:
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()
