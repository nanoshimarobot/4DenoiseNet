import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision

import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings
import time
import glob
import pprint
from typing import List

from dataset.kitti.utils import load_poses, load_calib

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_RESIDUAL = ['.npy']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_residual(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_RESIDUAL)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask,proj_labels

class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True,             # send ground truth?
               transform=False):
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.transform = transform
    
    ###################################
    self.dataset_size = 0
    self.index_mapping = {}
    dataset_index = 0
    ###################################

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = {}
    self.label_files = {}

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "snow_velodyne")
      label_path = os.path.join(self.root, seq, "snow_labels")
      
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))
      
      ######################################
      n_used_files = max(0, len(scan_files)) 
      for start_index in range(n_used_files):
        self.index_mapping[dataset_index] = (seq, start_index)
        dataset_index += 1
      self.dataset_size += n_used_files
      ######################################
      
      # extend list
      scan_files.sort()
      label_files.sort()

      #self.clean_scan_files[seq] = clean_scan_files
      self.scan_files[seq] = scan_files
      self.label_files[seq] = label_files
        
    print("Using {} scans from sequences {}".format(self.dataset_size,
                                                    self.sequences))

  def __getitem__(self, dataset_index):
    # Get sequence and start index
    seq, start_index = self.index_mapping[dataset_index]
    proj_full = torch.Tensor()
    pre_proj_full = torch.Tensor()
    # index is now looping from first scan in input sequence to current scan
    # for index in range(start_index, start_index + self.n_input_scans):
    for index in range(start_index, start_index + 1):
      # get item in tensor shape
      #clean_scan_file = self.clean_scan_files[seq][index]
      scan_file = self.scan_files[seq][index]
      
      if self.gt:
        label_file = self.label_files[seq][index]
  
      # open a semantic laserscan
      DA = False
      flip_sign = False
      rot = False
      drop_points = False
      jitter_x = 0
      jitter_y = 0
      jitter_z = 0
      if self.transform:
          if random.random() > 0.5:
              if random.random() > 0.5:
                  DA = True
                  jitter_x = random.uniform(-5,5)
                  jitter_y = random.uniform(-3,3)
                  jitter_z = random.uniform(-1,0)
              if random.random() > 0.5:
                  flip_sign = True
              if random.random() > 0.5:
                  rot = True
              drop_points = random.uniform(0, 0.5)
  
      if self.gt:
        scan = SemLaserScan(self.color_map,
                            project=True,
                            H=self.sensor_img_H,
                            W=self.sensor_img_W,
                            fov_up=self.sensor_fov_up,
                            fov_down=self.sensor_fov_down,
                            DA=DA,
                            flip_sign=flip_sign,
                            drop_points=drop_points,
                            jitter_x=jitter_x,
                            jitter_y=jitter_y,
                            jitter_z=jitter_z)
      else:
        scan = LaserScan(project=True,
                         H=self.sensor_img_H,
                         W=self.sensor_img_W,
                         fov_up=self.sensor_fov_up,
                         fov_down=self.sensor_fov_down,
                         DA=DA,
                         rot=rot,
                         flip_sign=flip_sign,
                         drop_points=drop_points,
                         jitter_x=jitter_x,
                         jitter_y=jitter_y,
                         jitter_z=jitter_z)

      # open and obtain (transformed) scan
      #scan.open_scan(scan_file, index_pose, current_pose, if_transform=self.transform_mod)
      scan.open_scan(scan_file)
      
      if self.gt:
        scan.open_label(label_file)
        # map unused classes to used classes (also for projection)
        scan.sem_label = self.map(scan.sem_label, self.learning_map)
        scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

      # make a tensor of the uncompressed data (with the max num points)
      unproj_n_points = scan.points.shape[0]
      unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
      unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
      unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
      unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
      unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
      unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
      if self.gt:
        unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
        unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
      else:
        unproj_labels = []
  
      # get points and labels
      proj_range = torch.from_numpy(scan.proj_range).clone()
      proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
      proj_remission = torch.from_numpy(scan.proj_remission).clone()
      proj_mask = torch.from_numpy(scan.proj_mask)
      if self.gt:
        proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
        proj_labels = proj_labels * proj_mask
      else:
        proj_labels = []
      proj_x = torch.full([self.max_points], -1, dtype=torch.long)
      proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
      proj_y = torch.full([self.max_points], -1, dtype=torch.long)
      proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        
      proj = torch.cat(
          [proj_range.unsqueeze(0).clone(),
           proj_xyz.clone().permute(2, 0, 1),
           proj_remission.unsqueeze(0).clone()])
      proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]

      proj_full = torch.cat([proj_full, proj])

      ###################################################
      # open previous scan
      if index > 0: pre_scan_file = self.scan_files[seq][index - 1]
      else: pre_scan_file = self.scan_files[seq][index]
      
      scan.open_scan(pre_scan_file)
      
      pre_proj_range = torch.from_numpy(scan.proj_range).clone()
      pre_proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
      pre_proj_remission = torch.from_numpy(scan.proj_remission).clone()
      
      pre_proj = torch.cat(
          [pre_proj_range.unsqueeze(0).clone(),
           pre_proj_xyz.clone().permute(2, 0, 1),
           pre_proj_remission.unsqueeze(0).clone()])
      pre_proj = (pre_proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]

      pre_proj_full = torch.cat([pre_proj_full, pre_proj])
      ######################################################
    
    proj_full = proj_full * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")

    # return
    return proj_full, pre_proj_full, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, \
           unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return self.dataset_size

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

class SemanticT4(Dataset):
    def __init__(
        self,
        augment,
        # color_map = [255, 0, 0],  # colors dict bgr (e.g 10: [255, 0, 0])
        # learning_map,  # classes to learn (0 to N-1 for xentropy)
        # learning_map_inv,  # inverse of previous (recover labels)
        # sensor,  # sensor to parse scans from
        # max_points=150000,  # max number of points present in dataset
        # gt=True,  # send ground truth?
        # transform=False,
    ):
        self.data_dir = "/home/toyozoshimada/perception_dataset/data/rainy_dataset_non_annotated_t4_format/t4_rainy"
        # self.pcd_data = sorted(
        #     glob.glob(os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "*.pcd.bin"))
        # )
        print("Loaded SemanticT4 loader")
        self.pcd_data = [
           os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00000.pcd.bin"),
           os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00010.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00020.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00030.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00040.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00050.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00060.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00070.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00080.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00090.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00100.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00110.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00120.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00130.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00140.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00150.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00160.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00170.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00180.pcd.bin"),
          #  os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "00190.pcd.bin"),
        ]

        cloud_dim = 6
        self.cloud_list: List[np.ndarray] = []
        self.index_mapping = {}
        self.dataset_size = 0
        dataset_index = 0
        self.augment = augment

        self.color_map = {
            0: [0, 255, 255],
            1: [255, 0, 0],
            2: [0, 255, 0],
        }
        self.sensor_img_H = 64
        self.sensor_img_W = 2048
        self.sensor_fov_up = 3
        self.sensor_fov_down = -25
        self.gt = True
        self.learning_map = {
            0: 0,
            1: 1,
            2: 2,
        }
        self.learning_map_inv = {
            0: 0,
            1: 1,
            2: 2,
        }
        self.sensor_img_means = torch.tensor([12.12, 10.88, 0.23, -1.04, 0.21], dtype=torch.float)
        self.sensor_img_stds = torch.tensor([12.32, 11.47, 6.91, 0.86, 0.16], dtype=torch.float)
        # self.learning_ignore

        cloud_d_type = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.uint8),
                ("return_type", np.uint8),
                ("channel", np.uint16),
                ("padding", np.float32),
                ("entity_id", np.uint32),
            ]
        )
        # 一旦時系列的に連続と仮定
        for path in self.pcd_data:
            print(path)
            raw_cloud = np.fromfile(path, dtype=cloud_d_type)
            self.cloud_list.append(raw_cloud)

    def scan_to_proj_data(self, scan: LaserScan | SemLaserScan, labels):
        proj_full = torch.Tensor()
        if self.gt:
            # print(f"Labels : {labels}")
            scan.set_label(labels.reshape((-1)))
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)
        self.max_points = 200000
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        proj = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ]
        )
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]

        proj_full = torch.cat([proj_full, proj])

        return (
            proj_full,
            proj_mask,
            proj_labels,
            unproj_labels,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points,
        )

    def cloud_to_proj_data(self, index):
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        jitter_x = 0.0
        jitter_y = 0.0
        jitter_z = 0.0
        if self.augment:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                    jitter_x = random.uniform(-5, 5)
                    jitter_y = random.uniform(-3, 3)
                    jitter_z = random.uniform(-1, 0)
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)  # bool to float

        if self.gt:
            scan = SemLaserScan(
                {0: [0, 255, 255], 1: [255, 0, 0], 2: [0, 255, 0]},
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
                DA=DA,
                drop_points=drop_points,
                jitter_x=jitter_x,
                jitter_y=jitter_y,
                jitter_z=jitter_z,
                flip_sign=flip_sign,
            )
        else:
            scan = LaserScan(
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
                DA=DA,
                rot=rot,
                flip_sign=flip_sign,
                drop_points=drop_points,
                jitter_x=jitter_x,
                jitter_y=jitter_y,
                jitter_z=jitter_z,
            )

        scan_x = self.cloud_list[index]["x"].astype(np.float32)
        scan_y = self.cloud_list[index]["y"].astype(np.float32)
        scan_z = self.cloud_list[index]["z"].astype(np.float32)
        points = np.vstack((scan_x, scan_y, scan_z)).T
        remissions = self.cloud_list[index]["intensity"].astype(np.uint8)
        labels = self.cloud_list[index]["entity_id"]
        # print(labels)
        if drop_points:
            drop_idx = np.random.randint(
                0, len(points) - 1, int(len(points) * drop_points)
            )
            points = np.delete(points, drop_idx, axis=0)
            remissions = np.delete(remissions, drop_idx)
            labels = np.delete(labels, drop_idx)

        scan.set_points(points, remissions)

        return self.scan_to_proj_data(scan, labels)

    def __getitem__(self, index):
        proj_full = torch.Tensor()
        pre_proj_full = torch.Tensor()

        # guard処理
        if index >= len(self.cloud_list) - 1:
            pass
        # print(f"index: {index}")
        (
            pre_proj_full,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.cloud_to_proj_data(index)

        (
            proj_full,
            proj_mask,
            proj_labels,
            unproj_labels,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points,
        ) = self.cloud_to_proj_data(index + 1)

        proj_full = proj_full * proj_mask.float()

        return (
            proj_full,
            pre_proj_full,
            proj_mask,
            proj_labels,
            unproj_labels,
            0,
            "",
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points,
        )

    def __len__(self):
      return len(self.cloud_list)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
            # do the mapping
        return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               split,             # split (train, valid, test)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=False):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.split = split
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    if self.split == 'train':
      # self.train_dataset = SemanticKitti(root=self.root,
      #                                    sequences=self.train_sequences,
      #                                    labels=self.labels,
      #                                    color_map=self.color_map,
      #                                    learning_map=self.learning_map,
      #                                    learning_map_inv=self.learning_map_inv,
      #                                    sensor=self.sensor,
      #                                    max_points=max_points,
      #                                    transform=True, # set to True to augment the data
      #                                    gt=self.gt)
      self.train_dataset = SemanticT4(True)
  
      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle_train, 
                                                     # shuffle=False, # set False to ensure sequential loading
                                                     num_workers=self.workers,
                                                     drop_last=True)
      assert len(self.trainloader) > 0
      self.trainiter = iter(self.trainloader)

      # self.valid_dataset = SemanticKitti(root=self.root,
      #                                    sequences=self.valid_sequences,
      #                                    labels=self.labels,
      #                                    color_map=self.color_map,
      #                                    learning_map=self.learning_map,
      #                                    learning_map_inv=self.learning_map_inv,
      #                                    sensor=self.sensor,
      #                                    max_points=max_points,
      #                                    gt=self.gt)
      self.valid_dataset = SemanticT4(True)

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.workers,
                                                     drop_last=True)
      assert len(self.validloader) > 0
      self.validiter = iter(self.validloader)

    # if self.split == 'valid':
    #   self.valid_dataset = SemanticKitti(root=self.root,
    #                                      sequences=self.valid_sequences,
    #                                      labels=self.labels,
    #                                      color_map=self.color_map,
    #                                      learning_map=self.learning_map,
    #                                      learning_map_inv=self.learning_map_inv,
    #                                      sensor=self.sensor,
    #                                      max_points=max_points,
    #                                      gt=self.gt)
  
    #   self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
    #                                                  batch_size=self.batch_size,
    #                                                  shuffle=False,
    #                                                  num_workers=self.workers,
    #                                                  drop_last=True)
    #   assert len(self.validloader) > 0
    #   self.validiter = iter(self.validloader)
    
    if self.split == 'test':
      if self.test_sequences:
        self.test_dataset = SemanticKitti(root=self.root,
                                          sequences=self.test_sequences,
                                          labels=self.labels,
                                          color_map=self.color_map,
                                          learning_map=self.learning_map,
                                          learning_map_inv=self.learning_map_inv,
                                          sensor=self.sensor,
                                          max_points=max_points,
                                          gt=False)
  
        self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      drop_last=True)
        assert len(self.testloader) > 0
        self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)

