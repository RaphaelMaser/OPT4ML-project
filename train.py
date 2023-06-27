import torch
from ray import air, tune
import ray
from ray.air.config import RunConfig, ScalingConfig
from ray.train.lightning import LightningTrainer, LightningConfigBuilder
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from data import CIFAR10DataModule
from model import SmallNetLightning

def train(batch_size=64, epochs=10, config=None, accelerator="cuda" if torch.cuda.is_available() else "cpu", num_workers=1, cwd=None, logger_name="my_model", train_batches_per_epoch=0.1, val_batches_per_epoch=0.2):
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
    
    data_module = CIFAR10DataModule(batch_size=batch_size, cwd=cwd)
    #logger = TensorBoardLogger(os.path.join(cwd,"tensorboard_logs"), name=logger_name)
    #logger = CSVLogger(os.path.join(cwd,"csv_logs"), name=logger_name)
    #tune_logger = TuneReportCallback(["train_loss, val_loss"], on="on_test_epoch_end")
    
    lightning_config = (
        LightningConfigBuilder()
        .module(cls=SmallNetLightning, config=config)
        .trainer(max_epochs=epochs, accelerator=accelerator, logger=None, callbacks=[RichProgressBar(leave=True)], limit_train_batches=train_batches_per_epoch, limit_val_batches=val_batches_per_epoch)
        .fit_params(datamodule=data_module)
        #.checkpointing(monitor="val_loss", save_top_k=0, mode="min")
        .build()
    )

    run_config = RunConfig(
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=1,
        #     checkpoint_score_attribute="val_loss",
        #     checkpoint_score_order="min",
        #  ),
    )

    scaling_config = ScalingConfig(
        num_workers=num_workers, use_gpu=True if accelerator=="cuda" else False, resources_per_worker={"CPU": 1, "GPU": 1 if accelerator=="cuda" else 0}
    )

    # Define a base LightningTrainer without hyper-parameters for Tuner
    lightning_trainer = LightningTrainer(
        scaling_config=scaling_config,
        run_config=run_config,
    )   

    tuner = tune.Tuner(
        lightning_trainer,
        param_space={"lightning_config": lightning_config},
        tune_config=tune.TuneConfig(
            max_concurrent_trials=1,
            metric="val_loss",
            mode="min",
            #num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            storage_path=cwd,
            name=f"tune_cifar10_{logger_name}",
        ),
    )
    results = tuner.fit()
    #best_result = results.get_best_result(metric="val_loss", mode="max")
    return results