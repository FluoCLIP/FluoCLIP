import argparse
import os
import os.path as osp
import pdb
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from fluoclip.runner.data_stain import RegressionDataModule
from fluoclip.runner.runner import Runner
from fluoclip.utils.logging import get_logger, setup_file_handle_for_all_logger

logger = get_logger(__name__)

# resume option command line
# --cfg-options resume.stage_num=1 \
# resume.ckpt_path="/path/to/your/checkpoint/last.ckpt"

def main(cfg: DictConfig):
    pl.seed_everything(cfg.runner_cfg.seed, True)
    output_dir = Path(cfg.runner_cfg.output_dir)
    setup_file_handle_for_all_logger(str(output_dir / "run.log"))

    stage1_callbacks = load_callbacks(output_dir, stage=1)
    stage2_callbacks = load_callbacks(output_dir, stage=2)
    loggers = load_loggers(output_dir, cfg)

    deterministic = True
    logger.info(f"`deterministic` flag: {deterministic}")
    # combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage1_cfg)

    if cfg.trainer_cfg.fast_dev_run is True:
        from IPython.core.debugger import set_trace
        set_trace()

    runner = None
    stage2_trainer = None
    regression_datamodule = RegressionDataModule(**OmegaConf.to_container(cfg.data_cfg))

    # --- Two-Stage Training Orchestration ---
    if not cfg.test_only:
        resume_stage = cfg.resume.stage_num
        resume_ckpt_path = cfg.resume.ckpt_path
        logger.info(f"resume_stage : {resume_stage}, resume_ckpt_path: {resume_ckpt_path}")
        stage1_ckpt_path = None # This variable will hold the Stage 1 checkpoint path
        if resume_stage == 1:
            logger.info(f"[RESUME] Stage 1 from {resume_ckpt_path} (weights+optimizer)")
            combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage1_cfg)

            stage1_runner = Runner(
                **OmegaConf.to_container(combined_cfg),
                stage_num=1
            )

            trainer_stage1_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage1_cfg.max_epochs})
            stage1_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage1_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage1_cfg),
            )
            stage1_trainer.fit(model=stage1_runner, datamodule=regression_datamodule, ckpt_path=resume_ckpt_path)

            stage1_ckpt_path = stage1_trainer.checkpoint_callback.best_model_path
            logger.info(f"Stage 1 finished. Best model saved at: {stage1_ckpt_path}")
            
            # Step 2: Stage 2 Training (Fine-tuning)
            start_ckpt_path = stage1_ckpt_path
            logger.info(f"Starting Stage 2: Fine-tuning from checkpoint: {start_ckpt_path}")

            combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage2_cfg)
            stage2_runner = Runner.load_from_checkpoint(checkpoint_path=start_ckpt_path, strict=False, map_location="cpu",**OmegaConf.to_container(combined_cfg), stage_num=2)

            if cfg.stage2_cfg.max_epochs:
                trainer_stage2_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage2_cfg.max_epochs})
            else:
                trainer_stage2_cfg = cfg.trainer_cfg
            
            stage2_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage2_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage2_cfg),
            )

            stage2_trainer.fit(
                model=stage2_runner,
                datamodule=regression_datamodule,
                ckpt_path=None,
            )
            logger.info("Stage 2 finished.")

        elif resume_stage==2:
            logger.info(f"[RESUME] Stage 2 from {resume_ckpt_path} (weights+optimizer)")
            combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage2_cfg)

            try:
                import torch
                sd = torch.load(resume_ckpt_path, map_location="cpu").get("state_dict", {})
                if not any("rank_embeds" in k for k in sd.keys()):
                    logger.warning("[RESUME] The checkpoint doesn't seem Stage-2 (no 'rank_embeds'). "
                                   "Full resume may fail if param groups differ. Use weights-only path instead.")
            except Exception as e:
                logger.warning(f"[RESUME] Could not inspect checkpoint: {e}")

            stage2_runner = Runner(
                **OmegaConf.to_container(combined_cfg),
                stage_num=2
            )

            if cfg.stage2_cfg.max_epochs:
                trainer_stage2_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage2_cfg.max_epochs})
            else:
                trainer_stage2_cfg = cfg.trainer_cfg

            stage2_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage2_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage2_cfg),
            )
            stage2_trainer.fit(
                model=stage2_runner,
                datamodule=regression_datamodule,
                ckpt_path=resume_ckpt_path,
            )
            logger.info("Stage 2 finished.")

        elif resume_stage is None:
            logger.info("Starting Stage 1: Training Stain-aware Prompts.")
            combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage1_cfg)

            stage1_runner = Runner(
                **OmegaConf.to_container(combined_cfg),
                stage_num=1
            ).cuda()

            trainer_stage1_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage1_cfg.max_epochs})
            stage1_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage1_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage1_cfg),
            )

            stage1_trainer.fit(model=stage1_runner, datamodule=regression_datamodule)
            
            stage1_ckpt_path = stage1_trainer.checkpoint_callback.best_model_path
            logger.info(f"Stage 1 finished. Best model saved at: {stage1_ckpt_path}")
        
            start_ckpt_path = stage1_ckpt_path

            logger.info(f"Starting Stage 2: Fine-tuning from checkpoint: {start_ckpt_path}")
            combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage2_cfg)
            # stage2_runner = Runner(
            #     **OmegaConf.to_container(combined_cfg),
            #     stage_num=2
            # ).cuda()
            stage2_runner = Runner.load_from_checkpoint(checkpoint_path=start_ckpt_path, strict=False,map_location="cpu",**OmegaConf.to_container(combined_cfg), stage_num=2)
            # Create a new Trainer for Stage 2
            if cfg.stage2_cfg.max_epochs :
                trainer_stage2_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage2_cfg.max_epochs})
            else : trainer_stage2_cfg = cfg.trainer_cfg
            
            stage2_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage2_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage2_cfg),
            )
            
            # Load the Stage 1 checkpoint to continue training
            # This will load the weights of all layers that were saved
            stage2_trainer.fit(
                model=stage2_runner, 
                datamodule=regression_datamodule, 
                ckpt_path=None,
            )
            logger.info("Stage 2 finished.")

    # Testing
    ckpt_paths = list((Path(output_dir) / f"stage2" / "ckpts").glob("*.ckpt"))
    if len(ckpt_paths) == 0:
        logger.info("zero shot")
        if stage2_trainer is None: 
            if cfg.stage2_cfg.max_epochs :
                trainer_stage2_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage2_cfg.max_epochs})
            stage2_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage2_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage2_cfg),
            )
        if runner is None:
            runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
        stage2_trainer.test(model=runner, datamodule=regression_datamodule)
        logger.info(f"End zero shot.")

    for ckpt_path in ckpt_paths:
        logger.info(f"Start testing ckpt: {ckpt_path}.")

        # no need to load weights in runner wrapper
        for k in cfg.runner_cfg.load_weights_cfg.keys():
            cfg.runner_cfg.load_weights_cfg[k] = None
        cfg.runner_cfg.ckpt_path = str(ckpt_path)

        combined_cfg = OmegaConf.merge(cfg.runner_cfg, cfg.stage2_cfg)
        if stage2_trainer is None: 
            if cfg.stage2_cfg.max_epochs :
                trainer_stage2_cfg = OmegaConf.merge(cfg.trainer_cfg, {"max_epochs": cfg.stage2_cfg.max_epochs})
            stage2_trainer = pl.Trainer(
                logger=loggers,
                callbacks=stage2_callbacks,
                deterministic=deterministic,
                **OmegaConf.to_container(trainer_stage2_cfg),
            )
        if runner is None:
            runner = Runner(**OmegaConf.to_container(combined_cfg), stage_num=2)

        runner = runner.load_from_checkpoint(str(ckpt_path), **OmegaConf.to_container(combined_cfg), stage_num=2)
        stage2_trainer.test(model=runner, datamodule=regression_datamodule)

        logger.info(f"End testing ckpt: {ckpt_path}.")


def load_loggers(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "tb_logger").mkdir(exist_ok=True, parents=True)
    (output_dir / "csv_logger").mkdir(exist_ok=True, parents=True)
    wandb_project_name = output_dir.parent.name

    loggers = []
    # tb_logger = pl_loggers.TensorBoardLogger(
    #     str(output_dir),
    #     name="tb_logger",
    # )
    loggers.append(
        pl_loggers.CSVLogger(
            str(output_dir),
            name="csv_logger",
        )
    )
    wandb_logger = pl_loggers.WandbLogger(
            name=wandb_project_name, 
            save_dir=str(output_dir), # Use save_dir for WandbLogger to specify where files like run logs are saved locally
            log_model=False,
            project="fluoclip_n", # Recommended: Specify your W&B project name
        )

    log_config_to_wandb(wandb_logger, config)
    loggers.append(wandb_logger)
    return loggers

def log_config_to_wandb(wandb_logger, config):
    runner_cfg = config["runner_cfg"]

    model_cfg = runner_cfg.get("model_cfg", {})
    model_params = flatten_dict(model_cfg, parent_key="model")

    optimizer_cfg = runner_cfg.get("optimizer_and_scheduler_cfg", {})
    loss_weights = runner_cfg.get("loss_weights", {})
    trainer_cfg = config.get("trainer_cfg", {})
    data_cfg = config.get("data_cfg", {})

    train_params = {
        **flatten_dict(optimizer_cfg, parent_key="optim"),
        **flatten_dict(loss_weights, parent_key="loss"),
        **flatten_dict(trainer_cfg, parent_key="trainer"),
        **flatten_dict(data_cfg, parent_key="data"),
        "seed": runner_cfg.get("seed", None),
    }

    stage1_cfg = flatten_dict(config.get("stage1_cfg", {}), parent_key="stage1")
    stage2_cfg = flatten_dict(config.get("stage2_cfg", {}), parent_key="stage2")

    wandb_logger.experiment.config.update(model_params, allow_val_change=True)
    wandb_logger.experiment.config.update(train_params, allow_val_change=True)
    wandb_logger.experiment.config.update(stage1_cfg, allow_val_change=True)
    wandb_logger.experiment.config.update(stage2_cfg, allow_val_change=True)


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten nested dicts (OmegaConf safe)."""
    items = {}
    if isinstance(d, DictConfig):
        d = dict(d)  # convert to plain dict

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, (dict, DictConfig)):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def load_callbacks(output_dir, stage: int, monitor="val_mae_max_metric", mode="min"):
    """
    Saves to: {output_dir}/stage{stage}/ckpts
    Example: load_callbacks("tmp", stage=1)
    """
    stage_dir = Path(output_dir) / f"stage{stage}"
    ckpt_dir  = stage_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # filename uses the monitored key; double braces keep {} in f-strings
    filename = f"{{epoch:02d}}-{{{monitor}:.4f}}"

    cb = ModelCheckpoint(
        monitor=monitor,
        dirpath=str(ckpt_dir),
        filename=filename,
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode=mode,
        save_weights_only=False,
    )
    return [cb]


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    return output_dir


def get_version(path: Path):
    versions = path.glob("version_*")
    return len(list(versions))


def parse_cfg(args, instantialize_output_dir=True):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    seed = args.seed if args.seed is not None else cfg.runner_cfg.seed
    ckpt_path = Path(cfg.ckpt_path if args.ckpt_path is None else args.ckpt_path)

    if str(ckpt_path) == 'None':
        cli_cfg = OmegaConf.create(
            dict(
                config=args.config,
                test_only=args.test_only,
                runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
                trainer_cfg=dict(fast_dev_run=args.debug),
            )
        )
    else:
        cli_cfg = OmegaConf.create(
            dict(
                config=args.config,
                test_only=args.test_only,
                runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
                trainer_cfg=dict(fast_dev_run=args.debug, max_epochs=100),
                ckpt_path=str(ckpt_path),
            )
        )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None,
        help="Set the path of ckpt file")
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = parse_cfg(args, instantialize_output_dir=True)

    logger.info("Start.")
    main(cfg)
    logger.info("End.")
