import argparse
import tqdm
import importlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

WANDB_ENTITY = 'fliu'
WANDB_PROJECT = 'vis_libero'

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False

# create TF dataset
dataset_name = 'libero_dataset'
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
# change to your data directory
ds = tfds.load(dataset_name, data_dir='/shared/fangchen/libero', split='train')

# visualize episodes
for i, episode in enumerate(ds.take(184)):
    images = []
    for step in episode['steps']:
        images.append(step['observation']['image'].numpy()[::-1])

    image_strip_1 = np.concatenate(images[-20:][::4], axis=1)
    image_strip_2 = np.concatenate(images[:20][::4], axis=1)
    caption = step['task'].numpy().decode() + ' (temp. downsampled 4x)'

    if render_wandb:
        wandb.log({f'image_end_{i}': wandb.Image(image_strip_1, caption=caption)})
        wandb.log({f'image_beg_{i}': wandb.Image(image_strip_2, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip_1)
