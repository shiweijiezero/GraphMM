import os

import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data_loader import PLDataModule
from wrapper import Wrapper
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning


def main():
    config_dir = 'config.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)
    os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

    dataset = PLDataModule(batch_size=config["batch"])
    dataset.setup()
    # for features,labels in dataset.train_dataloader():
    #     print(features.shape)
    #     print(labels.shape)
    #     break

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="../../data/log",
        version=None,
        name='lightning_logs'
    )

    early_stopping = pytorch_lightning.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=3,
                                                               mode='min')

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=2,
        save_last=True,
        dirpath="./data/save_model",
    )

    model = Wrapper(config, minibatch_size=config["batch"])
    # trainer = Trainer(precision=16,
    #                   max_epochs=config["epoch"],
    #                   auto_scale_batch_size="power",
    #                   logger=tb_logger,
    #                   accelerator="gpu", devices=2, auto_select_gpus=True,
    #                   strategy="ddp",
    #                   callbacks=[checkpoint_callback,early_stopping]
    #                   )
    # # trainer.tune(model)
    # trainer.fit(model, dataset)
    # # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

    trainer_test = Trainer(precision=16,
                      max_epochs=1,
                      # auto_scale_batch_size="power",
                      logger=tb_logger,
                      accelerator="gpu", devices=1, auto_select_gpus=True,
                      )
    trainer_test.test(model=model,ckpt_path="./data/save_model/last.ckpt", dataloaders=dataset)


if __name__ == '__main__':
    main()
