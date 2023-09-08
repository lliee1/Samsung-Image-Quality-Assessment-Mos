from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SumMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd


class ManiqaModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_pred_epoch = []
        self.train_labels_epoch = []
        self.val_pred_epoch = []
        self.val_labels_epoch = []

        self.mos_ls = []
        self.image_name_ls = []
        self.comments_ls = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()



    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        img, score = batch['d_img_org'], batch['score'].squeeze()
        pred = self.forward(img)
        pred = self.sig(pred)
        loss = self.criterion(pred, score)
        return loss, pred, score

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, pred, score = self.model_step(batch)
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = score.data.cpu().numpy()
        self.train_pred_epoch = np.append(self.train_pred_epoch, pred_batch_numpy)
        self.train_labels_epoch = np.append(self.train_labels_epoch, labels_batch_numpy)
    
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        
        rho_s, _ = spearmanr(np.squeeze(self.train_pred_epoch), np.squeeze(self.train_labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(self.train_pred_epoch), np.squeeze(self.train_labels_epoch))
        self.log("train/PLCC", rho_s, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/SRCC", rho_p, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_pred_epoch = []
        self.train_labels_epoch = []
        self.train_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred, score = self.model_step(batch)
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = score.data.cpu().numpy()
        self.val_pred_epoch = np.append(self.val_pred_epoch, pred_batch_numpy)
        self.val_labels_epoch = np.append(self.val_labels_epoch, labels_batch_numpy)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)



    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        rho_s, _ = spearmanr(np.squeeze(self.val_pred_epoch), np.squeeze(self.val_labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(self.val_pred_epoch), np.squeeze(self.val_labels_epoch))
        self.log("val/metric", rho_s+rho_p, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_pred_epoch = []
        self.val_labels_epoch = []


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        img, img_name = batch
        pred = self.forward(img)

        for mos in pred:
            self.mos_ls.append(round(mos,11))
    
        for name in img_name:
            self.image_name_ls.append(name)

        for _ in range(len(img_name)):
            self.comments_ls.append('Nice image')

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        
        submit_df = pd.DataFrame(self.image_name_ls, columns=['img_name'])
        submit_df.insert(1, 'mos', self.mos_ls)
        submit_df.insert(2, 'comments', self.comments_ls)
        submit_df.to_csv('/root/dacon/data/submit_maniqa.csv', mode='w', index=False)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/metric",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
