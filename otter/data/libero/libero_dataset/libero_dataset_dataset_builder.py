"""libero_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from otter.data.libero.libero_dataset.libero_dataset import LIBERODataset


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for libero_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(libero_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
                'steps':
                    tfds.features.Dataset({
                     'observation': tfds.features.FeaturesDict({
                        'proprio': tfds.features.Tensor(shape=(6,), dtype=tf.float32),
                        'image': tfds.features.Image(shape=(256, 256, 3)),
                        'prev_action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                        # 'image_1': tfds.features.Image(shape=(256, 256, 3)),
                       }),
                    'task': tfds.features.Text(),
                    'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    'extrinsics': tfds.features.Tensor(shape=(4,4), dtype=tf.float32),
                    'intrinsics': tfds.features.Tensor(shape=(3,3), dtype=tf.float32),
                }),
            }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # TODO(libero_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    # train_dir = '/home/ravenhuang/LIBERO/libero/datasets/train'
    train_dir = '/home/fangchen/libero_test/'
    return {
        'train': self._generate_examples(LIBERODataset(train_dir)),
    }

  def _generate_examples(self, dataset):
    """Yields examples."""
    idx = 0
    for traj in dataset:
        # traj = dataset[0]
        # traj_idx = 0
        for task_traj in traj:
            yield idx, {
                'steps': task_traj,
            }
            idx += 1
