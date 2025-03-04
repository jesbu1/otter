import math
import sys
from typing import Iterable
import os 

import torch
import torch.nn as nn
from . import misc

from otter.util.args import ExperimentConfig
from torch.utils.tensorboard import SummaryWriter
from otter.policy.otter import OTTER
from otter.util.misc import NativeScalerWithGradNormCount as NativeScaler

def train(
    model: OTTER, 
    train_data_loader: Iterable, 
    val_data_loader: Iterable, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler : torch.optim.lr_scheduler,
    loss_scaler : NativeScaler,
    log_writer : SummaryWriter = None, 
    args : ExperimentConfig = None,
):
    
    total_steps = args.trainer_cfg.num_steps
    save_freq_steps = args.shared_cfg.save_every
    val_every = args.shared_cfg.val_every
    num_val_steps = args.shared_cfg.num_val_steps

    model.train()
    optimizer.zero_grad() # Clear gradients only during training

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 10

    accum_iter = args.trainer_cfg.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, dataset_item in enumerate(metric_logger.log_every_step(train_data_loader, print_freq, total_steps)):
        # forward pass and compute loss
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**dataset_item)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss["loss"]
            else:
                loss_dict = {}
        
        loss_value = loss.item()
        loss_value_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Normalize loss
        loss /= accum_iter
        loss_value_dict = {k: v / accum_iter for k, v in loss_value_dict.items()}
        
        # perform backward pass, update gradients and optimizer
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # update learning rate
        lr_scheduler.step()

        # clear gradients every accum_iter steps
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Log metrics
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Reduce loss and log to wandb
        loss_value_reduce = loss_value
        
        # Log to wandb
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('train_loss', loss_value_reduce, data_iter_step)
            log_writer.add_scalar('lr', lr, data_iter_step)

        # validation 
        if data_iter_step > 0 and (data_iter_step) % val_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_step in range(num_val_steps):
                    val_dataset_item = next(val_data_loader)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        val_loss = model(**val_dataset_item)
                    if isinstance(val_loss, dict):
                        val_loss = val_loss["loss"]
                    val_loss_value = val_loss.item()
                    if val_step % print_freq == 0:
                        print(f"Validation Step {val_step} / {num_val_steps}, loss: {val_loss_value}")
                    val_loss += val_loss_value
                val_loss = val_loss / num_val_steps
                metric_logger.update(val_loss=val_loss_value)
                if log_writer is not None:
                    log_writer.add_scalar('val_loss', val_loss_value, data_iter_step)
            model.train()
            optimizer.zero_grad()

        # save checkpoint
        if data_iter_step % save_freq_steps == 0 or data_iter_step >= total_steps:
            print("Step: {}. Synchronizing logger and save checkpoint.".format(data_iter_step))
            print("Averaged stats:", metric_logger)
            # Save checkpoint
            if args.logging_cfg.output_dir and misc.is_main_process():
                checkpoint_path = os.path.join(args.logging_cfg.output_dir, f'checkpoint_{data_iter_step}.pt')
                if args.distributed: 
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, checkpoint_path)

        if data_iter_step >= total_steps:
            print("Reached total steps, stopping training")
            break