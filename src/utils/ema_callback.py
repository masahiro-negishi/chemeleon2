from lightning.pytorch.callbacks import Callback
from torch_ema import ExponentialMovingAverage


class EMA(Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_train_batch_end(self, trainer, pl_module, *_):
        self.ema.update(pl_module.parameters())

    def on_validation_start(self, trainer, pl_module):
        self.ema.store()
        self.ema.copy_to(pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, ckpt):
        ckpt["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, ckpt):
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)
        self.ema.load_state_dict(ckpt["ema_state"])
