# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation 
from PIL import Image


# export DEX_YCB_DIR=/home/ravenhuang/hbrl/dexycb
_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_MANO_MCP_IND = [0, 1, 5, 9, 13, 17] #the first is the wrist
_MANO_THUMB_TIP_IND = 4
_MANO_INDEX_TIP_IND = 8

_BOP_EVAL_SUBSAMPLING_FACTOR = 4

SHM_PATH = '/dev/shm/hbrl_data/dexycb_imgs/'
LOCAL_RESIZE_PATH = '/home/ravenhuang/hbrl_data/dexycb_imgs/'
target_size = (320, 240)
#label format:
#pose_y: object pose, [num_obj, 3, 4]
#pose_m: human hand pose MANO, last 3 is translation
#joint_3d: human 3d joint in camera frame
#joint_2d: human 2d joint in image space

def estimate_plane(ycb_pose_array):
  """
  ycb_pose_array: np.array of shape (num_objects, 3) of translations
  Returns the quaternion and centroid of the plane passing through all YCB Objects.
  """
  CoG_pos = ycb_pose_array
  # print("CoG_pos: ", CoG_pos)

  # Create the design matrix

  A = np.column_stack((CoG_pos[:, 0],
                        CoG_pos[:, 1],
                        np.ones(CoG_pos.shape[0])))
  
  # Solve the least square problem to find the coeff of the plane
  V = np.linalg.lstsq(A, CoG_pos[:, 2], rcond=None)[0]

  # Compute the normal of the plane
  normal = np.array([V[0], V[1], -1])
  normal = normal / np.linalg.norm(normal)

  # Compute the quaternion
  angle = np.arccos(np.dot(normal, np.array([0, 0, 1])))

  # Compute axis angle representation
  w = np.cos(angle / 2)
  axis = np.sin(angle / 2) * normal
  x,y,z = axis[0], axis[1], axis[2]

  # Compute the centroid of the YCB Objects projected on the plane
  centroid = np.mean(CoG_pos, axis=0)
  quat = np.array([x, y, z, w], dtype=np.float32)

  # Translate the centroid 0.2m below the YCB objects
  # centroid += 0.4 * normal
  rot = Rotation.from_quat(quat).as_matrix()
  mat = np.eye(4)
  mat[:3, :3] = rot
  mat[:3, 3] = centroid

  return mat

def get_hand_ori(hand_joint_3d):
  '''
  estimate the hand orientation from the 3d joints in the world frame
  this function fit a plane for the palm using the mcp translation
  '''
  #extract all the mcp and wrist
  mcp_points = hand_joint_3d[_MANO_MCP_IND] #shape (6,3)
  palm_T = estimate_plane(mcp_points)
  return palm_T

def approx_parrallel_jaw_gripper(hand_joint_3d):
  '''
  use the thumb and index tip to estimate the gripper distance
  '''
  thumb_tip_pos = hand_joint_3d[_MANO_THUMB_TIP_IND]
  index_tip_pos = hand_joint_3d[_MANO_INDEX_TIP_IND]
  distance = np.linalg.norm(thumb_tip_pos - index_tip_pos)
  return distance

def create_intri_mat(intr):
  fx,fy,ppx,ppy = intr['fx'], intr['fy'], intr['ppx'], intr['ppy']
  intri_mat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
  return intri_mat

class DexYCBDataset():
  """DexYCB dataset."""
  ycb_classes = _YCB_CLASSES
  mano_joints = _MANO_JOINTS
  mano_joint_connect = _MANO_JOINT_CONNECT

  def __init__(self):
    """Constructor.
    """
    
    assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._data_dir = os.environ['DEX_YCB_DIR']
    self._calib_dir = os.path.join(self._data_dir, "calibration")
    self._model_dir = os.path.join(self._data_dir, "models")

    self._color_format = "color_{:06d}.jpg"
    self._resize_color_format = "resize_color_{:06d}.jpg"
    self._depth_format = "aligned_depth_to_color_{:06d}.png"
    self._label_format = "labels_{:06d}.npz"
    self._h = 480
    self._w = 640

    self._obj_file = {
        k: os.path.join(self._model_dir, v, "textured_simple.obj")
        for k, v in _YCB_CLASSES.items()
    }
    
    subject_ind = [0, 1, 2 ,3, 4, 5, 6, 7, 8, 9]
    serial_ind  = [0, 1, 2, 3, 4, 5, 6, 7]
    sequence_ind = list(range(100))

    self._subjects = [_SUBJECTS[i] for i in subject_ind]

    self._serials = [_SERIALS[i] for i in serial_ind]
    self._intrinsics = []
    for s in self._serials:
      intr_file = os.path.join(self._calib_dir, "intrinsics",
                               "{}_{}x{}.yml".format(s, self._w, self._h))
      with open(intr_file, 'r') as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
      intr = intr['color']
      self._intrinsics.append(intr)
      
    self._sequences = []
    self._mapping = []
    self._ycb_ids = []
    self._ycb_grasp_ind = []
    self._mano_side = []
    self._mano_betas = []
    offset = 0
    
    for n in self._subjects:
      seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
      seq = [os.path.join(n, s) for s in seq]
      assert len(seq) == 100
      seq = [seq[i] for i in sequence_ind]
      self._sequences += seq
      for i, q in enumerate(seq):
        meta_file = os.path.join(self._data_dir, q, "meta.yml")
        with open(meta_file, 'r') as f:
          meta = yaml.load(f, Loader=yaml.FullLoader)
          

        extr_file = self._calib_dir + "/extrinsics_" + meta[
        'extrinsics'] + "/extrinsics.yml"
        
        m = [offset+i,len(self._serials), meta['num_frames'], extr_file]
        self._mapping.append(m)
        self._ycb_ids.append(meta['ycb_ids'])
        self._ycb_grasp_ind.append(meta['ycb_grasp_ind'])
        self._mano_side.append(meta['mano_sides'][0])
        mano_calib_file = os.path.join(self._data_dir, "calibration",
                                       "mano_{}".format(meta['mano_calib'][0]),
                                       "mano.yml")
        with open(mano_calib_file, 'r') as f:
          mano_calib = yaml.load(f, Loader=yaml.FullLoader)
        self._mano_betas.append(mano_calib['betas'])
        
      offset += len(seq)
    self._mapping = np.vstack(self._mapping)

  def __len__(self):
    return len(self._mapping)

  def __getitem__(self, idx):
    s, l_c, n_f, extr_file = self._mapping[idx]
    s, l_c, n_f = int(s), int(l_c), int(n_f)

    with open(extr_file, 'r') as f:
      extr = yaml.load(f, Loader=yaml.FullLoader)
    all_extrinsics = extr['extrinsics']
    april_tag = np.array(all_extrinsics['apriltag']).reshape(3,4) # this is the table frame in world frame
    table_T = np.eye(4)
    table_T[:3, :3] = april_tag[:3, :3]
    table_T[:3, 3] = april_tag[:3, 3]
    
    samples = []
    for c in range(l_c):
      sample_c = []
      serial = self._serials[c]
      d = os.path.join(self._data_dir, self._sequences[s], serial)
      
      smh_d = d.replace(self._data_dir, SHM_PATH)
      if not os.path.isdir(smh_d):
        os.makedirs(smh_d)
        
      local_resize_d = d.replace(self._data_dir, LOCAL_RESIZE_PATH)
      if not os.path.isdir(local_resize_d):
        os.makedirs(local_resize_d)
        
      extr_c = all_extrinsics[serial]
        
      extrinsics = np.array(extr_c).reshape(3,4)
      extr_t = np.eye(4)
      extr_t[:3, :3] = extrinsics[:3, :3]
      extr_t[:3, 3] = extrinsics[:3, 3]
      
      T_table_camera = np.linalg.inv(table_T) @ extr_t
      
      for f in range(n_f):
        color_file = os.path.join(d, self._color_format.format(f))
        label_file = os.path.join(d, self._label_format.format(f))
        
        label = np.load(label_file)
        mano_pose = label['pose_m']
        joint_3d = label['joint_3d'][0]
        joint_2d = label['joint_2d']
        wrist_pos = joint_3d[0,0]
        
        if np.all(wrist_pos==-1): #hand not visible
          continue
        
        ycb_ids = self._ycb_ids[s]
        ycb_grasp_ind = self._ycb_grasp_ind[s]
        
        object_poses = label['pose_y'][ycb_grasp_ind] #in camera frame
        obj_c = np.eye(4)
        obj_c[:3,:3] = object_poses[:3,:3]
        obj_c[:3,3] = object_poses[:3,3]
        grasp_obj_t = T_table_camera @ obj_c
        
        grasped_obj = ycb_ids[ycb_grasp_ind]
        name_obj = ' '.join(_YCB_CLASSES[grasped_obj].split('_')[1:])
        language_instruction = f'pick up the {name_obj}'
        
        joint3d_table = (np.tile(T_table_camera[:3,:3],(len(joint_3d),1,1)) @ joint_3d[...,None]).squeeze() + T_table_camera[:3,3]
        hand_ori = get_hand_ori(joint3d_table)
        hand_pose = np.eye(4)
        hand_pose[:3,:3] = hand_ori[:3,:3]
        hand_pose[:3,3] = joint3d_table[0]
        gripper_distance = approx_parrallel_jaw_gripper(joint3d_table)
        
        resize_path = os.path.join(smh_d, self._resize_color_format.format(f))
        local_resize_img_path = os.path.join(local_resize_d, self._resize_color_format.format(f))
        if not os.path.isfile(local_resize_img_path):
          image = Image.open(color_file)
          img_resized = image.resize(target_size)
          img_resized.save(local_resize_img_path)
        else:                        
          img_resized = Image.open(local_resize_img_path)
        
        # image_array = np.array(image)
        
        sample ={
          'observation':{
            "object_pose": grasp_obj_t.astype(np.float32), #object poses in table frame
            "hand_pose": hand_pose.astype(np.float32),
            "gripper_distance": gripper_distance,
            "joint_3d": joint3d_table.astype(np.float32),
            # "image": color_file,
            "image": np.array(img_resized),
            },
          "task": language_instruction,
          'extrinsics': T_table_camera.astype(np.float32),
          'intrinsics': create_intri_mat(self._intrinsics[c]).astype(np.float32),
        }
        
        sample_c.append(sample)
        
      samples.append(sample_c)
        
      
    return samples
  


  @property
  def data_dir(self):
    return self._data_dir

  @property
  def h(self):
    return self._h

  @property
  def w(self):
    return self._w

  @property
  def obj_file(self):
    return self._obj_file

  def get_bop_id_from_idx(self, idx):
    """Returns the BOP scene ID and image ID given an index.

    Args:
      idx: Index of sample.

    Returns:
      scene_id: BOP scene ID.
      im_id: BOP image ID.
    """
    s, c, f = map(lambda x: x.item(), self._mapping[idx])
    scene_id = s * len(self._serials) + c
    im_id = f
    return scene_id, im_id


