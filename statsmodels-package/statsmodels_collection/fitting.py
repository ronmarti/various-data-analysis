from os import PathLike
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def fit(module, train_dataloader,
        max_epochs=1000,
        patience=10,
        min_delta=1e-4,
        verbose=True,
        enable_checkpointing=True,
        enable_logger=True,
        checkpoint_path: PathLike = None):
    """Runs pytorch lightning training with earlystopping and constructs default
    trainer for you.

    Args:
        module ([type]): pylightning module to be trained.
        train_dataloader ([type]): [description]
        max_epochs (int, optional): [description]. Defaults to 1000.
        patience (int, optional): [description]. Defaults to 10.
        min_delta ([type], optional): [description]. Defaults to 1e-4.
        verbose (bool, optional): [description]. Defaults to True.
        enable_checkpointing (bool, optional): [description]. Defaults to True.
        enable_logger (bool, optional): [description]. Defaults to True.
        checkpoint_path (PathLike, optional): path to save the final model 
            in the form of checkpoint. Defaults to None.

    Returns:
        PathLike: checkpoint_path
    """     
    callbacks = [
        EarlyStopping(
            monitor='train_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode='min',
            check_on_train_epoch_end=True
        )
    ]

    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            save_top_k=1,
            mode='min',
        )
        callbacks += [checkpoint_callback]

    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        checkpoint_callback=enable_checkpointing,
        logger=enable_logger
    )
    trainer.fit(module, train_dataloader)
    if checkpoint_path is not None:
        trainer.save_checkpoint(checkpoint_path)

    return checkpoint_path