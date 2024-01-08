from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from speechsep.cli import Args
from speechsep.model import Demucs
from speechsep.util import center_trim


class LitDemucs(LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        self.model = Demucs(args)

        self.metric_sisdr = ScaleInvariantSignalDistortionRatio()
        self.metric_pesq = PerceptualEvaluationSpeechQuality(
            8000, "nb", n_processes=args["num_workers"]
        )
        self.metric_stoi = ShortTimeObjectiveIntelligibility(8000, False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = l1_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, sisdr, pesq, stoi = self._shared_eval_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict({"val_sisdr": sisdr, "val_pesq": pesq, "val_stoi": stoi}, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, sisdr, pesq, stoi = self._shared_eval_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log_dict({"test_sisdr": sisdr, "test_pesq": pesq, "test_stoi": stoi}, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = l1_loss(y_pred, y)

        # Calculate average metrics over all channels and examples
        sisdr = self.metric_sisdr(y_pred, y)
        try:
            pesq = self.metric_pesq(y_pred, y)
        except Exception:
            # Errors can occur when PESQ is not computable (NoUtterancesError) or when using multiprocessing
            pesq = 0.0
        stoi = self.metric_stoi(y_pred, y)

        return loss, sisdr, pesq, stoi

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=self.args["weight_decay"])
        optimizer_config = {
            "optimizer": optimizer,
        }

        # Configure learning rate scheduler
        if (
            self.args["cosine_anneal_period"] is not None
            and self.args["reduce_on_plateau_metric"] is not None
        ):
            raise ValueError(
                "Only one of cosine_anneal_period and reduce_on_plateau_metric can be set."
            )

        if self.args["cosine_anneal_period"] is not None:
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            optimizer_config["lr_scheduler"] = CosineAnnealingLR(
                optimizer, steps_per_epoch * self.args["cosine_anneal_period"]
            )
            print(
                f"Using cosine annealing LR scheduler with period {self.args['cosine_anneal_period']} epochs."
            )
        elif self.args["reduce_on_plateau_metric"] is not None:
            optimizer_config["lr_scheduler"] = {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": self.args["reduce_on_plateau_metric"],
            }
            print(
                f"Using reduce on plateau LR scheduler with metric {self.args['reduce_on_plateau_metric']}."
            )

        return optimizer_config
