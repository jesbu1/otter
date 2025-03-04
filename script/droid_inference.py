import contextlib
import signal
from collections import deque
from PIL import Image
from typing import Optional, List
import numpy as np
import numpy as np
import mediapy as media
import os
import time
import torch
import tyro
import datetime
import tqdm
import cv2

from droid.data_processing.timestep_processing import TimestepProcesser
from droid.robot_env import RobotEnv

from otter.policy.otter_interface import OtterInference
from otter.dataset.utils import action_10d_to_7d

def get_success_rate() -> float:
    success: str | float | None = None
    while not isinstance(success, float):
        success = input(
            "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
        )
        if success == "y":
            success = 1.0
        elif success == "n":
            success = 0.0
        elif success == "":
            success = 1.0
        success = float(success) / 100
        if not (0 <= success <= 1):
            print(f"Success must be a number in [0, 100] but got: {success * 100}")
    return success

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

def save_pil_images_to_video(
    images: List[Image.Image],
    output_path: str,
    fps: int = 30
) -> None:
    """
    Convert a list of PIL images to a video file using mediapy.
    
    Args:
        images: List of PIL Image objects
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    
    Raises:
        ValueError: If images list is empty or images have different sizes
    """
    if not images:
        raise ValueError("Image list is empty")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert PIL images to numpy array
    frames = []
    for img in images:
        # Convert PIL image to numpy array
        frame = np.array(img)
        frames.append(frame)
    
    # Stack frames into a single numpy array
    video_array = np.stack(frames)
    
    # Write video using mediapy
    media.write_video(output_path, video_array, fps=fps)

class OTTERRolloutPolicy:
    def __init__(
            self, 
            model_ckpt_folder : str, 
            ckpt_id : int,
            action_exec_horizon : int, 
            receeding_horizon_control : bool = False,
            language_prompt : Optional[str] = None,
        ):
        policy = OtterInference(
            model_ckpt_folder=model_ckpt_folder,
            ckpt_id=ckpt_id,
        )
        self.policy = policy
        print("Loaded policy from checkpoint")
        self.model_id = model_ckpt_folder.split("/")[-1]
        self.timestep = 0
        self.action_exec_horizon = action_exec_horizon
        action_pred_horizon = policy.model.action_horizon
        assert self.action_exec_horizon <= action_pred_horizon, f"Action exec horizon must be less than or equal to {action_pred_horizon}"
        self.last_call = None
        self.curr_call = time.time()
        
        self.action_dim = policy.model.action_dim
        self.traj_len = policy.model.seq_length

        print("Context window size: ", self.traj_len)
        self.rollout_idx = 0
        self.language_prompt = language_prompt  # the prompt to use for the model

        # action queue for receeding horizon control or temporal ensemble
        self.action_queue = deque([],maxlen=self.action_exec_horizon)

        self.receeding_horizon_control = receeding_horizon_control

    def set_language_prompt(self, language_prompt : str):
        assert isinstance(language_prompt, str), "Language prompt must be a string"
        print("Setting language prompt to: ", language_prompt)
        self.language_prompt = language_prompt

    def start_episode(self, demo_idx=0):
        self.timestep = 0
        print("Creating tasks: ", self.language_prompt)
        self.action_queue.clear()
        self.policy.reset()
        print("Starting episode...")
        self.rollout_idx += 1
    
    def __call__(self, ob : dict):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """ 
        self.timestep += 1

        # vision inputs
        img_primary = (ob["camera/image/varied_camera_1_left_image"]*255).astype(np.uint8).transpose(1, 2, 0)
        img_wrist = (ob["camera/image/hand_camera_left_image"]*255).astype(np.uint8).transpose(1, 2, 0)
        img_primary = Image.fromarray(img_primary).convert('RGB')
        img_wrist = Image.fromarray(img_wrist).convert('RGB')

        images = {
            "image_primary": img_primary,
            "image_wrist": img_wrist
        }
        proprio = ob['robot_state/cartesian_position']
        gripper = ob['robot_state/gripper_position']    

        action = self.policy(
            images = images, 
            text = self.language_prompt, 
            proprio = proprio, 
            gripper = gripper
        ) # np.ndarray of shape (action_chunk_size, action_dim)

        if self.receeding_horizon_control:
            # Receeding horizon control start
            if len(self.action_queue) == 0:
                self.action_queue = deque(action[:self.action_exec_horizon])
            action = self.action_queue.popleft()
            # receeding horizon control ends
        else:
            # temporal emsemble start
            new_actions = deque(action[:self.action_exec_horizon])
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), self.action_dim))
            
            k = 0.05
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()
            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()
            action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            # temporal ensemble ends

        action = action_10d_to_7d(action)
        action[-1] = action[-1] > 0.5
        return action

class PolicyWrapperRobomimic:
    def __init__(
        self, 
        policy : OTTERRolloutPolicy, 
        action_space : str = "cartesian_position", 
        gripper_action_space : str = "position",
        video_output_dir : Optional[str] = None,
    ):
        self.policy = policy
        timestep_filtering_kwargs=dict(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
        )
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=True,
            to_tensor=True,
            augment=False,
        )
        self.timestep_processor = TimestepProcesser(
            ignore_action=True, 
            **timestep_filtering_kwargs, 
            image_transform_kwargs=image_transform_kwargs
        )
        if video_output_dir is not None:
            self.record = True
            # generate output dir where video is saved 
            self.video_output_dir = os.path.join(video_output_dir, self.policy.model_id)
            os.makedirs(self.video_output_dir, exist_ok=True)
            print("Video output dir: ", self.video_output_dir)
            self.video_cache = []
        else:
            self.record = False

    def start_episode(self, language_prompt : str):
        print("Starting episode...")
        if self.record:
            # empty cache
            self.video_cache = []
        self.start_timestep = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        self.policy.set_language_prompt(language_prompt)
        self.policy.start_episode()

    def end_episode(self):
        print("Ending episode...")
        if self.record:
            prompt = self.policy.language_prompt.replace(" ", "_")
            video_path = os.path.join(
                self.video_output_dir, 
                f"{prompt}_{self.start_timestep}.mp4"
            )
            print("Saving video to: ", video_path)
            save_pil_images_to_video(self.video_cache, video_path)
            print("Video is saved!")
            self.video_cache = []

    def forward(self, observation):
        timestep = {"observation": observation}
        processed_timestep = self.timestep_processor.forward(timestep)
        obs = {
            "robot_state/cartesian_position": np.array(observation["robot_state"]["cartesian_position"]),
            "robot_state/joint_positions": np.array(observation["robot_state"]["joint_positions"]),
            "robot_state/gripper_position": np.array([observation["robot_state"]["gripper_position"]]), # wrap as array, raw data is single float
            "camera/image/hand_camera_left_image": processed_timestep["observation"]["camera"]["image"]["hand_camera"][0],
            "camera/image/hand_camera_right_image": processed_timestep["observation"]["camera"]["image"]["hand_camera"][1],
            "camera/image/varied_camera_1_left_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][0], # (3, 320, 180)
            "camera/image/varied_camera_1_right_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][1],
        }
        for k in obs:
            obs[k] = np.array(obs[k])

        if self.record:
            img_primary = (obs["camera/image/varied_camera_1_left_image"]*255).astype(np.uint8).transpose(1, 2, 0)
            wrist_camera = (obs["camera/image/hand_camera_left_image"]*255).astype(np.uint8).transpose(1, 2, 0)
            img = np.concatenate([img_primary, wrist_camera], axis=0)
            # show the image 
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            img = Image.fromarray(img).convert('RGB')
            self.video_cache.append(img)

        action = self.policy(obs)
        return action

def main(
    model_ckpt_folder : str, # path to the model checkpoint
    ckpt_id : int, # id of the checkpoint (e.g. 60000)
    action_exec_horizon : int = 12, # number of actions to execute in the rollout
    receeding_horizon_control : bool = True, # whether to use receeding horizon control
    video_output_dir : str = "video_outputs", # directory to save the video outputs
    max_timesteps : int = 600, # maximum number of timesteps to run the rollout
):
    ACTION_SPACE = "cartesian_position"
    GRIPPER_ACTION_SPACE = "position"
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(320, 180), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(320, 180), resize_func="cv2"),
    )
    env = RobotEnv(
        action_space=ACTION_SPACE, 
        gripper_action_space=GRIPPER_ACTION_SPACE,
        camera_kwargs=camera_kwargs,
    )
    print("Created the droid env!")
    # we evaluate the model here 
    rollout_policy = OTTERRolloutPolicy(
        model_ckpt_folder, 
        ckpt_id,
        action_exec_horizon, 
        receeding_horizon_control=receeding_horizon_control,
    )
    wrapped_policy = PolicyWrapperRobomimic(
        rollout_policy, 
        ACTION_SPACE, 
        GRIPPER_ACTION_SPACE,
        video_output_dir=video_output_dir, 
    )
    print("Created the rollout policy!")

    num_trials = 0
    last_instruction = None
    while True: 
        instruction = input("Enter instruction: ")
        if last_instruction is not None and instruction == "":
            print("Using last instruction: ", last_instruction)
            instruction = last_instruction
        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        wrapped_policy.start_episode(instruction)
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = env.get_observation()
                with prevent_keyboard_interrupt():
                    # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                    action = wrapped_policy.forward(curr_obs)
                comp_time = time.time() - start_time
                sleep_left = (1 / env.control_hz) - comp_time
                if sleep_left > 0:
                    time.sleep(sleep_left)
                start_time = time.time()
                # if action is all zeros, skip! 
                if not np.any(action):
                    continue
                env.step(action)
            except KeyboardInterrupt:
                break
        wrapped_policy.end_episode()
        success_rate = get_success_rate()
        num_trials += 1
        print("Current trajectory success rate: ", success_rate)
        print("Overall success rate: ", success_rate/num_trials)
        if input("Do one more eval? (enter y or n) ").lower() not in ["y", ""]:
            break
        last_instruction = instruction
        env.reset()

if __name__ == '__main__':
    tyro.extras.set_accent_color("yellow")
    tyro.cli(main)