import dataclasses
from typing import Literal, Optional, Tuple, Union, List
import enum
import pathlib
from otter.data.dataset import hoi4d_restructure, dexycb_restructure, droid_restructure, icrt_restructure

import tyro

@dataclasses.dataclass
class DatasetConfig:
    """Configuration for dataset handling and preprocessing"""
    
    # Dataset parameters
    dataset_kwargs: List[dict] = dataclasses.field(default_factory=list)  # Changed from None
    sample_weights: Optional[List[float]] = dataclasses.field(default_factory=list)  # Changed from None
    
    # Trajectory transformation parameters
    traj_transform_kwargs: dict = dataclasses.field(default_factory=dict)  # Changed from None
    frame_transform_kwargs: dict = dataclasses.field(default_factory=dict)  # Changed from None
    
    # Data loading parameters
    shuffle_buffer_size: int = 1600
    traj_transform_threads: int = 48
    traj_read_threads: int = 48
    prefetch_num_batches: int = 10
    
    # Logging configuration
    log_vis_data: Optional[List[str]] = dataclasses.field(default_factory=list)  # Changed from None

    def __post_init__(self):
        if not self.dataset_kwargs:  # Changed condition since it's now an empty list
            self.dataset_kwargs = [
                dict(
                    name='icrt_pickplace', # 473
                    data_dir='dataset/icrt_pickplace/1.0.0',
                    shuffle=True, action_normalization_mask=[True] * 9 + [False], skip_norm=True,
                    action_proprio_normalization_type='normal',
                    proprio_noise=0.01,
                    restructure = icrt_restructure,
                ),
                dict(
                    name='icrt_stack', #109
                    data_dir='dataset/icrt_stack_mul/1.0.0',
                    shuffle=True, action_normalization_mask=[True] * 9 + [False], skip_norm=True,
                    action_proprio_normalization_type='normal',
                    proprio_noise=0.01,
                    restructure = icrt_restructure,
                ),
                dict(
                    name='icrt_0926', #150
                    data_dir='dataset/icrt_pickplace_1/1.0.0',
                    shuffle=True, action_normalization_mask=[True] * 9 + [False], skip_norm=True,
                    action_proprio_normalization_type='normal',
                    proprio_noise=0.01,
                    restructure = icrt_restructure,
                ),
                dict(
                    name = 'icrt_poke', #185
                    data_dir = 'dataset/icrt_poke/1.0.0',
                    shuffle=True, action_normalization_mask=[True]*9+[False], skip_norm=True,
                    action_proprio_normalization_type = 'normal',
                    proprio_noise = 0.01,
                    restructure = icrt_restructure,
                ),
                dict(
                    name = 'icrt_drawer', #167
                    data_dir = 'dataset/icrt_drawer/1.0.0',
                    shuffle=True, action_normalization_mask=[True]*9+[False], skip_norm=True,
                    action_proprio_normalization_type = 'normal',
                    proprio_noise = 0.01,
                    restructure = icrt_restructure,
                ),
                dict(
                    name = 'icrt_pour', #101
                    data_dir = 'dataset/icrt_pour/1.0.0',
                    shuffle=True, action_normalization_mask=[True]*9+[False], skip_norm=True,
                    action_proprio_normalization_type = 'normal',
                    proprio_noise = 0.01,
                    restructure = icrt_restructure,
                ),
            ]
            
        if not self.traj_transform_kwargs:  # Changed condition
            self.traj_transform_kwargs = {
                "goal_relabeling_strategy": None,
                "task_augment_strategy": None,
            }
            
        if not self.frame_transform_kwargs:  # Changed condition
            self.frame_transform_kwargs = {
                "image_dropout_prob": 0.0,
                "image_augment_kwargs": {
                    "primary": {
                        "random_brightness": [0.2],
                        "random_contrast": [0.8, 1.2],
                        "random_saturation": [0.8, 1.2],
                        "random_hue": [0.1],
                        "augment_order": [
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ]
                    }
                },
                "num_parallel_calls": 400,
            }

        if not self.sample_weights:
            self.sample_weights = [1.5, 1.5, 1.5, 2.5, 4.5, 6]    
        

@dataclasses.dataclass
class ModelConfig: 
    # CLIP model architecture to use for vision encoding
    clip_model: str = "ViT-L/14"
    
    # num_readouts for attention pooling
    num_readouts: int = 4
    
    # Input dimension for proprioception data
    proprio_input_dim: int = 10
    
    # Hidden dimension for proprioception processing
    proprio_hidden_dim: int = 256
    
    # Output dimension for proprioception features
    proprio_output_dim: int = 64
    
    # Output dimension for text pooling layer
    text_pooling_output_dim: int = 128

    # number of tokens used for CLIP 
    first_k_tokens : int = 15
    
    # Output dimension for vision pooling layer (combined value for all cameras)
    vision_pooling_output_dim: int = 512
    
    # Number of attention heads in pooling layers
    pooling_heads: int = 8
    
    # Number of pooling layers
    pooling_layers: int = 2
    
    # Dimension of action space
    action_dim: int = 10

    # max position embeddings
    max_position_embeddings : int = 32

    # Number of transformer layers in the model
    transformer_layers: int = 8
    
    # Number of attention heads in transformer layers
    transformer_heads: int = 8
    
    # Hidden dimension size in transformer layers
    transformer_dim: int = 768

    # Transformer expansion factor more consistent with paper
    transformer_expansion_factor: int = 1

    # attention dropout probability
    attention_probs_dropout_prob : int = 0.0

    # dropout probability
    hidden_dropout_prob : int = 0.0

    # only true text token are used for attention pooling 
    pool_true_text : bool = True

@dataclasses.dataclass
class ActionDecoderConfig:
    # hidden size
    hidden_size : int = 1024
    
    # number of transformer blocks
    num_hidden_layers : int = 6

@dataclasses.dataclass
class OptimizerConfig: 
    # weight decay (default: 0.01) 
    weight_decay : float = 0.01

    # learning rate (absolute lr)
    lr : Optional[float] = 3e-4

    # lower lr bound for cyclic schedulers that hit 0
    min_lr : float = 0.0 

@dataclasses.dataclass
class TrainerConfig:
    # compile the model
    compile : bool = False

    # warmup steps
    warmup_steps : int = 2000

    # number of training steps 
    num_steps : int = int(8e4)

    # Accumulate gradient iterations (for increasing the effective batch size under memory constraints)
    accum_iter : int = 1

    # pin memory for dataloader
    pin_memory : bool = True

    # number of workers for dataloader 
    num_workers : int = 20 
    
@dataclasses.dataclass
class SharedConfig:
    # Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus
    batch_size : int = 64

    # Use 6DoF Rotation 
    rot_6d : bool = True 

    # number of frames in a sequence 
    seq_length : int = 12

    # seed for random number generators
    seed : int = 0
    
    # start epoch 
    start_epoch : int = 0

    # frequency of saving checkpoint (steps)
    save_every : int = 5000

    # val frequency (steps)
    val_every : int = 1000

    # number of steps to run validation for
    num_val_steps : int = 50

    # resume from checkpoint 
    resume : Optional[str] = None 

    # Number of cameras 
    camera_keys : Tuple[str] = ("image_primary", "image_wrist")

    # Number of future timesteps to predict actions for
    action_horizon: int = 12
    
    # use delta action
    use_delta_action : bool = True

    # scale action with calculated action statistics (json file)
    scale_action : Optional[str] = None

    # image size (Dino: need to change to 518)
    image_size : int = 224

@dataclasses.dataclass
class LoggingConfig:
    # path where to save, empty for no saving
    output_dir: str = "./output"

    # path where to save tensorboard logs 
    log_dir : Optional[str] = None 

    # log name (for wandb)
    log_name : Optional[str] = None

@dataclasses.dataclass
class ExperimentConfig: 
    # Dataset configuration
    dataset_cfg: DatasetConfig

    # Model configuration
    model_cfg: ModelConfig

    # Optimizer configuration
    optimizer_cfg: OptimizerConfig

    # Shared configuration
    shared_cfg: SharedConfig

    # Logging configuration 
    logging_cfg: LoggingConfig

    # trainer configuration
    trainer_cfg: TrainerConfig

    # train or eval 
    train : bool = True

    # number of distributed processes (required by torch distributed)
    world_size: int = 1

    # local rank of the process (required by torch distributed)
    local_rank: int = -1

    # distributed training on the optimizer (required by torch distributed)
    dist_on_itp: bool = False

    # distributed training url (required by torch distributed)
    dist_url: str = 'env://'

    # device to use for training / testing (required by torch distributed)
    device : str = "cuda"

    # load config. instead of using command line arguments, load from a config file
    load_config: Optional[str] = None


@dataclasses.dataclass
class InferenceConfig:
    # path to model checkpoint folder
    model_ckpt_folder: str

    # checkpoint index
    ckpt_id : int

if __name__ == "__main__": 
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)