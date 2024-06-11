# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""
 logs the current learing rate each epoch
"""

import pytorch_lightning


class LRLoggingCallback(pytorch_lightning.Callback):
    def on_train_epoch_start(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule):
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log("current_lr", current_lr, on_epoch=True)
