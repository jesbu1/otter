import time
from otter.policy.otter_interface import OtterInference
import numpy as np
from PIL import Image 

def test_otter_inference(
    model_ckpt_folder : str, 
    ckpt_id : int, 
):
    # Test _preprocess_images
    print("testing otter model")
    otter_inference = OtterInference(model_ckpt_folder=model_ckpt_folder, ckpt_id=ckpt_id)
    print("otter model loaded") 
    
    image_resolution = otter_inference.args.shared_cfg.image_size
    images = {
        'image_primary': Image.new('RGB', (image_resolution, image_resolution)),
        'image_wrist': Image.new('RGB', (image_resolution, image_resolution))
    }
    
    for t in range(50):
        random_proprio = np.random.rand(6)
        random_gripper = np.random.rand(1)
        task = "text"

        out = otter_inference(
            images, task, random_proprio, random_gripper
        )
        print("Step: ", t)
        print("Action shape: ", out.shape)
        print("First action: ", out[0])
    
    otter_inference.reset()
    print("reset done") 

    start = time.time()
    for t in range(50):
        random_proprio = np.random.rand(6)
        random_gripper = np.random.rand(1)
        task = "text"

        out = otter_inference(
            images, task, random_proprio, random_gripper
        )
        print("Step: ", t)
        print("Action shape: ", out.shape)
        print("First action: ", out[0])
    end = time.time()
    print("Policy inference frequency: ", 50/(end-start))

if __name__ == "__main__":
    test_otter_inference(
        model_ckpt_folder = "/home/mfu/checkpoints/250211_2059",
        ckpt_id = 65000
    )