import os
from typing import Iterator, Tuple, Any
import glob
import tensorflow_datasets as tfds
from PIL import Image
import h5py
import numpy as np

SIZE=(256, 256)
def resize_image(image):
    # img = Image.fromarray(image).resize(SIZE, Image.Resampling.LANCZOS)
    img = Image.fromarray(image).resize(SIZE, Image.BICUBIC)
    return np.array(img)

class LIBERODataset():
    def __init__(self, data_dir):
        """Constructor.
        """

        self._data_dir = data_dir
        # count all files in data dir
        self._files = glob.glob(os.path.join(self._data_dir, '*.hdf5'))
        # self._mapping = []
        # for f in self._files:
        #     # load h5 file
        #     data = h5py.File(f, 'r')['data']
        #     demo_ids = list(data.keys())
        #     for demo_id in demo_ids:
        #         self._mapping.append((f, demo_id))

        print('Found {} files'.format(len(self._files)))
        # print('First file:', self._mapping[0])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        file_name = self._files[idx]
        data = h5py.File(file_name, 'r')['data']
        language_instruction = file_name.split('/')[-1].split('.')[0]
        language_instruction = language_instruction.split('_demo')[0]
        # if first character is capital, then find the keyword 'SCENE'
        if language_instruction[0].isupper():
            language_instruction = language_instruction.split('SCENE')[1]
            # remove the number and _ after SCENE
            language_instruction = language_instruction[2:]
        # replace _ with space
        language_instruction = language_instruction.replace('_', ' ')
        demo_ids = list(data.keys())
        print('Found {} demos in {}!!!'.format(len(demo_ids), language_instruction))
        all_episodes = []
        for demo_id in demo_ids:
            example = data[demo_id]
            episode = []
            for i in range(len(example['obs']['agentview_rgb'])):
                proprio = np.concatenate([example['obs']['ee_pos'][i], example['obs']['ee_ori'][i]])
                if i == 0:
                    prev_action = np.zeros((7,))
                else:
                    prev_action = example['actions'][i-1]
                observation = {
                    'proprio': proprio.astype(np.float32),
                    'image': resize_image(example['obs']['agentview_rgb'][i].astype(np.uint8)),
                    'prev_action': prev_action.astype(np.float32),
                    # 'image_1': resize_image(example['obs']['eye_in_hand_rgb'][i].astype(np.uint8)),
                }
                episode.append({
                    'observation': observation,
                    'action': example['actions'][i].astype(np.float32),
                    'task': language_instruction,
                    'extrinsics': np.zeros((4,4)).astype(np.float32),
                    'intrinsics': np.zeros((3,3)).astype(np.float32),
                })
            all_episodes.append(episode)

        return all_episodes
