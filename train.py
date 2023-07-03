import torch
from ray import air, tune
import ray
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from data import CIFAR10DataModule
from model import SmallNetLightning
from data import CIFAR10DataModule
from model import SmallNet
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_mnist_tune(config, num_epochs, batch_size, cwd, limit_train_batches, limit_test_batches, accelerator, enable_checkpointing):
    model = SmallNetLightning(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                ["val_loss", "train_loss"],
                on="train_epoch_end"),
            ],
        
        limit_train_batches=limit_train_batches,
        limit_test_batches=limit_test_batches,
        accelerator=accelerator,
        enable_checkpointing=enable_checkpointing,
        )
    trainer.fit(model, CIFAR10DataModule(batch_size=batch_size, cwd=cwd))
    
    
def train(batch_size=64, epochs=50, config=None, resources_per_trial={"CPU": 7, "GPU": 1 if torch.cuda.is_available() else 0}, cwd=None, logger_name="my_model", train_batches_per_epoch=0.1, val_batches_per_epoch=0.2, enable_checkpointing=False):
    '''
    batch_size: batch size for training
    config: dictionary containing the optimizer and its parameters
    accelerator: "cuda" or "cpu"
    num_workers: number of parallel trials to run (each trial uses 1 cpu core and 1 gpu core, if available)
    cwd: current working directory
    logger_name: name of the directory (used for tensorboard and csv logs)
    train_batches_per_epoch: fraction of training data to use per epoch
    val_batches_per_epoch: fraction of validation data to use per epoch
    '''
    torch.manual_seed(42)


    trainable = tune.with_resources(train_mnist_tune, resources_per_trial)
    tuner = tune.Tuner(
        tune.with_parameters(trainable,
                            num_epochs=epochs,
                            batch_size=batch_size,
                            cwd=cwd,
                            limit_train_batches=train_batches_per_epoch, 
                            limit_test_batches=val_batches_per_epoch, 
                            accelerator="gpu" if torch.cuda.is_available() else "cpu", 
                            enable_checkpointing=enable_checkpointing,
                            ),
        param_space=config,
        tune_config=tune.TuneConfig(
            max_concurrent_trials=20,
            metric="val_loss",
            mode="min",
            #num_samples=num_samples,
        ),
        run_config=RunConfig(
            storage_path=cwd,
            name=f"tune_cifar10_{logger_name}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
            ),
        ),
    )
    results = tuner.fit()
    return results