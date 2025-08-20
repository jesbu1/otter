#!/bin/bash

#SBATCH --job-name=train_otter
#SBATCH --output=slurm_out/train_otter_%j.out
#SBATCH --error=slurm_out/train_otter_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=28

conda run -n otter --no-capture-output /bin/bash -c "TF_FORCE_GPU_ALLOW_GROWTH=true torchrun \
    --nproc_per_node=2 \
    --master_port=1255 \
    script/train.py \
    --logging-cfg.log-name otter_bridge \
    --logging-cfg.output-dir output/otter_bridge \
    --shared-cfg.action-horizon 10 \
    --shared-cfg.camera-keys image_primary \
    --dataset-cfg.sample-weights 1.0 \
    --model-cfg.action-dim 7 \
    --model-cfg.proprio-input-dim 7 \
    --trainer-cfg.num_workers 24 \
    --shared-cfg.batch-size 320"

#CUDA_VISIBLE_DEVICES=0 python script/train.py \
#    --logging-cfg.log-name otter_bridge_debug \
#    --logging-cfg.output-dir output/otter_bridge_debug \
#    --shared-cfg.action-horizon 10 \
#    --shared-cfg.camera-keys image_primary \
#    --shared-cfg.batch-size 128 \
#    --model-cfg.action-dim 7 \
#    --model-cfg.proprio-input-dim 7 \
#    --dataset-cfg.sample-weights 1.0 \
#    --trainer-cfg.num_workers 0
#