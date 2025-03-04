"""libero_dataset dataset."""

import tensorflow_datasets as tfds
from otter.data.libero.libero_dataset.libero_dataset_dataset_builder import Builder
import dlimp as dl


class LiberoDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for libero_dataset dataset."""
  # TODO(libero_dataset):
  DATASET_CLASS = Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
    data_dir = "/home/ravenhuang/tensorflow_datasets/libero_dataset/1.0.0/"
    builder = tfds.builder_from_directory(builder_dir=data_dir)
    full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)

