import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import wandb
import tyro
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import yaml 
import queue
import threading
from typing import Dict, Any

from otter.util.args import ExperimentConfig
from otter.policy.otter import OTTER
from otter.util import misc
from otter.util.misc import NativeScalerWithGradNormCount as NativeScaler
from otter.util.lr_sched import get_scheduler
from otter.util.engine import train

# Import dataset handling functions
from otter.data.dataset import make_interleaved_dataset

class TFDatasetWrapper:
    """Wrapper to convert TF dataset to PyTorch format with background prefetching"""
    def __init__(self, tf_dataset, device, buffer_size=10):
        self.tf_dataset = tf_dataset
        self.device = device
        self.iterator = iter(tf_dataset)
        self.buffer_size = buffer_size
        
        # Create a queue for prefetched batches
        self.queue = queue.Queue(maxsize=buffer_size)
        
        # Start prefetch thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        # Wait for initial buffer to fill
        while self.queue.qsize() < buffer_size:
            continue
            
    def _convert_batch(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TF tensors to PyTorch tensors"""
        return {
            'images': {
                'image_primary': torch.from_numpy(item['observation']['image_primary'].numpy()).to(self.device, non_blocking=True),
                'image_wrist': torch.from_numpy(item['observation']['image_wrist'].numpy()).to(self.device, non_blocking=True),
            },
            'text': [i[0].numpy().decode('UTF-8') for i in item['task']['language_instruction']],
            'proprio': torch.from_numpy(item['observation']['proprio'].numpy()).to(self.device, non_blocking=True),
            'gt_actions': torch.from_numpy(item['action'].numpy()).to(self.device, non_blocking=True)
        }
    
    def _prefetch_worker(self):
        """Background thread to continuously prefetch and convert batches"""
        while True:
            try:
                item = next(self.iterator)
                batch = self._convert_batch(item)
                self.queue.put(batch, block=True)
            except Exception as e:
                print(f"Prefetch worker error: {e}")
                # Recreate iterator if needed (shouldn't happen with infinite dataset)
                self.iterator = iter(self.tf_dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.queue.get(block=True)

    def get_next(self):
        """Alternative to __next__ if you prefer explicit method calls"""
        return next(self)


def main(args: ExperimentConfig):
    # spawn is needed to initialize vision augmentation within pytorch workers
    torch.multiprocessing.set_start_method('spawn')
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))


    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    world_size = misc.get_world_size()
    print("world size: ", world_size)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_dir = args.logging_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Create model
    model = OTTER(
        model_config=args.model_cfg,
        shared_config=args.shared_cfg,
    )

    # Resume from checkpoint if specified
    if args.shared_cfg.resume:
        model = misc.load_state_dict_flexible(model, torch.load(args.shared_cfg.resume, map_location='cpu'))
        
    # Move model to device
    model = model.to(device)

    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Compile model for better performance
    if args.trainer_cfg.compile:
        model = torch.compile(model)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable parameter percentage: {trainable_params/total_params*100:.2f}%")

    # add action prediction horizon to dataset_kwargs 
    for dataset_kwargs in args.dataset_cfg.dataset_kwargs:
        dataset_kwargs['action_horizon'] = args.shared_cfg.action_horizon
    
    # add sequence length to traj_transform_kwargs 
    args.dataset_cfg.traj_transform_kwargs['subsample_length'] = args.shared_cfg.seq_length

    # add resize_size to frame_transform_kwargs
    args.dataset_cfg.frame_transform_kwargs['resize_size'] = (args.shared_cfg.image_size, args.shared_cfg.image_size)

    train_data = make_interleaved_dataset(
        dataset_kwargs_list=args.dataset_cfg.dataset_kwargs,
        sample_weights=args.dataset_cfg.sample_weights,
        train=True,
        shuffle_buffer_size=args.dataset_cfg.shuffle_buffer_size,
        traj_transform_kwargs=args.dataset_cfg.traj_transform_kwargs,
        frame_transform_kwargs=args.dataset_cfg.frame_transform_kwargs,
        batch_size=args.shared_cfg.batch_size // world_size,
    )
    val_data = make_interleaved_dataset(
        dataset_kwargs_list=args.dataset_cfg.dataset_kwargs,
        sample_weights=args.dataset_cfg.sample_weights,
        train=False,
        shuffle_buffer_size=200,
        traj_transform_kwargs=args.dataset_cfg.traj_transform_kwargs,
        frame_transform_kwargs=args.dataset_cfg.frame_transform_kwargs,
        batch_size=args.shared_cfg.batch_size // world_size,
    )

    # Wrap datasets
    # try train_data.iterator(prefetch=FLAGS.config.dataset.prefetch_num_batches), instead 
    train_loader = TFDatasetWrapper(train_data, device, buffer_size=args.dataset_cfg.prefetch_num_batches)
    val_loader = TFDatasetWrapper(val_data, device, buffer_size=args.dataset_cfg.prefetch_num_batches)

    # Initialize optimizer
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer_cfg.lr)
    loss_scaler = NativeScaler()

    # Initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=args.trainer_cfg.warmup_steps,
        num_training_steps=args.trainer_cfg.num_steps,
    )

    # Set up wandb logging
    if misc.is_main_process() and args.logging_cfg.log_name is not None:
        wandb.init(
            entity="otter",
            project="otter",
            name=args.logging_cfg.log_name,
            config=args,
            sync_tensorboard=True
        )

    # Set up tensorboard logging
    if misc.is_main_process() and args.logging_cfg.log_dir is not None:
        log_writer = SummaryWriter(log_dir=args.logging_cfg.log_dir)
    else:
        log_writer = None

    # Training loop
    print(f"Start training for {args.trainer_cfg.num_steps} steps")
    
    # Training phase
    train(
        model=model,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args
    )

if __name__ == '__main__':
    args = tyro.cli(ExperimentConfig)
    
    if args.load_config is not None: 
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader) 

    # creating the output directory and logging directory 
    if args.logging_cfg.log_name is not None: 
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # dump the args into a yaml file 
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    
    main(args)