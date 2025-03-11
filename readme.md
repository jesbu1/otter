# 🦦 OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction
by <a href="https://qingh097.github.io/">Huang Huang*</a>, <a href="https://fangchenliu.github.io/">Fangchen Liu*</a>, <a href="https://max-fu.github.io">Letian Fu*</a>, <a href="https://scholar.google.com/citations?user=9bt2Z5QAAAAJ&hl=en">Tingfan Wu</a>, <a href="https://www.mustafamukadam.com/">Mustafa Mukadam</a>, <a href="https://people.eecs.berkeley.edu/~malik/">Jitendra Malik</a>, <a href="https://goldberg.berkeley.edu">Ken Goldberg</a>, <a href="https://people.eecs.berkeley.edu/~pabbeel/">Pieter Abbeel</a> at UC Berkeley and Meta (*equal contribution).

[[Paper](http://arxiv.org/abs/2503.03734)] | [[Project Page](https://ottervla.github.io/)]

This repo contains the official re-implementation for *Otter: A Vision-Language-Action Model with Text-Aware Feature Extraciton*. The experiments in the paper are based on the original repo [here](https://github.com/Max-Fu/otter) implemented in [Jax](https://github.com/google/jax). 

Further information please contact <a href="https://qingh097.github.io/">Huang Huang</a>, <a href="https://fangchenliu.github.io/">Fangchen Liu</a>, <a href="https://max-fu.github.io">Letian Fu</a>, or post an issue on Github!

## Todos 
- [OpenCLIP](https://github.com/mlfoundations/open_clip) integration to allow training and inference with more powerful CLIP models.

## Updates  
- 2025-03-05: Initial release. 

## Setup
```bash
# create conda env
conda create -n otter python=3.10 -y
conda activate otter
# install torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# download repo 
git clone https://github.com/Max-Fu/otter.git
cd otter
pip install -e .
```

## Inference
We provide a simple interface for the OTTER model. For more details, please refer to the [otter interface](otter/policy/otter_interface.py).
```python
from otter.policy.otter_interface import OtterInference
from PIL import Image
import numpy as np

policy = OtterInference(
    model_ckpt_folder : str = "path/to/model/checkpoint",
    ckpt_id : int = 60000,
)

image_primary : Image.Image = ...
image_wrist : Image.Image = ...

# action is a numpy array of shape (action_horizon, action_dim)
action = policy(
    images = {
        "image_primary" : image_primary,
        "image_wrist" : image_wrist
    }, 
    text : str = ..., # language prompt
    proprio : np.ndarray = ..., # proprioception (6,)
    gripper : np.ndarray = ..., # gripper position (1,)
)
... 
# reset the policy's cache upon finishing a rollout 
policy.reset()
```
We additionally provide a script for rolling out the OTTER model in the DROID environment. For more details, please refer to the [droid inference script](script/droid_inference.py).
```bash
python script/droid_inference.py \
    --model-ckpt-folder path/to/model/checkpoint \
    --ckpt-id 60000 
```

## Dataset
We host the OTTER dataset on Hugging Face. They are TFDS to support pre-training on [Open X-Embodiment](https://robotics-transformer-x.github.io/). Alternatively, there is a converted LeRobot version of the dataset [here](https://huggingface.co/datasets/mlfu7/pi0_conversion) to fine-tune [Pi0](https://github.com/Physical-Intelligence/openpi), which uses joint positions for proprioception and joint velocities for action. The fine-tuning scripts are provided [here](https://github.com/Max-Fu/openpi). 
```bash
# first install huggingface-cli
pip install -U "huggingface_hub[cli]"
# download the datasets
mkdir -p dataset
pushd dataset
huggingface-cli download mlfu7/icrt_pour --repo-type dataset --local-dir .
huggingface-cli download mlfu7/icrt_drawer --repo-type dataset --local-dir .
huggingface-cli download mlfu7/icrt_poke --repo-type dataset --local-dir .
huggingface-cli download mlfu7/icrt_pickplace_1 --repo-type dataset --local-dir .
huggingface-cli download mlfu7/icrt_stack_mul_tfds --repo-type dataset --local-dir .
huggingface-cli download mlfu7/icrt_pickplace --repo-type dataset --local-dir .
popd
```

## Model Training 
We use the following command to train the OTTER model. We support multi-GPU training on a single node.
```bash 
TF_FORCE_GPU_ALLOW_GROWTH=true torchrun --nproc_per_node=2 --master_port=1255 script/train.py --logging-cfg.log-name <log_name> --logging-cfg.output-dir <output_dir> --shared-cfg.batch-size 128
```
To change the dataset paths and their subsampling ratios, please refer to the [training args](otter/util/args.py).
To see all the available options, 
```bash 
python script/train.py --help
```

## Visualization
We provide a script for visualizing the CLIP's visual patch feature's cosine similarity with the text features. For more detail, please refer to the [script](script/clip_visualization.py).
```bash
python script/clip_visualization.py --text "pour from the orange cup into the pink bowl" --image asset/droid_image.png 
```
You can also visualize with more powerful CLIP models with [OpenCLIP](https://github.com/mlfoundations/open_clip). A good combination we find is the ViT-L-14 `datacomp_xl_s13b_b90k` version. 
```bash 
python script/open_clip_visualization.py --text "pour from the orange cup into the pink bowl" --image asset/droid_image.png 
```

## License
This project is under the Apache 2.0 license. See [LICENSE](LICENSE.txt) for details.

## Acknowledgement
The code is based on [Octo](https://octo-models.github.io/), [MAE](https://github.com/facebookresearch/mae), [CrossMAE](https://github.com/TonyLianLong/CrossMAE), [ICRT](https://github.com/Max-Fu/icrt), [OpenPi](https://github.com/Physical-Intelligence/openpi), [DROID](https://droid-dataset.github.io/).

## Citation 
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@article{huang2025otter,
    title={Otter: A Vision-Language-Action Model with Text-Aware Feature Extraciton}, 
    author={Huang Huang and Fangchen Liu and Letian Fu and Tingfan Wu and Mustafa Mukadam and Jitendra Malik and Ken Goldberg and Pieter Abbeel},
    journal={arXiv preprint arXiv:2503.03734},
    year={2025}
}
```
