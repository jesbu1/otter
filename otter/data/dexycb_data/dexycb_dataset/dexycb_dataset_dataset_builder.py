"""dexycb_dataset dataset."""

import tensorflow_datasets as tfds
from dex_ycb_custom import DexYCBDataset
import tensorflow as tf


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dexycb_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(dexycb_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
                'steps': 
                    tfds.features.Dataset({
                     'observation': tfds.features.FeaturesDict({
                        'object_pose': tfds.features.Tensor(shape=(4,4), dtype=tf.float32),
                        'hand_pose': tfds.features.Tensor(shape=(4,4), dtype=tf.float32),
                        'gripper_distance': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'joint_3d': tfds.features.Tensor(shape=(21,3), dtype=tf.float32),
                        'image': tfds.features.Image(shape=(240, 320, 3)),
                       }),
                    'task': tfds.features.Text(),
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

    # TODO(dexycb_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(DexYCBDataset()),
    }

  def _generate_examples(self, dataset):
    """Yields examples."""
    # TODO(dexycb_dataset): Yields (key, example) tuples from the dataset
    idx = 0
    for traj in dataset:
      # traj = dataset[0]
      # traj_idx = 0
      for cam_traj in traj:
        yield idx, {
              'steps': cam_traj,
          }
        idx += 1
        