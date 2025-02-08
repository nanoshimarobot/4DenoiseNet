#!/usr/bin/env python3
import glob
import pprint
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../../"))

import numpy as np
import torch
from typing import List
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import tqdm
import random
import copy


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
        self.pcd_data = sorted(
            glob.glob(os.path.join(self.data_dir, "data", "LIDAR_CONCAT", "*.pcd.bin"))
        )

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
        self.sensor_img_means = [12.12, 10.88, 0.23, -1.04, 0.21]
        self.sensor_img_stds = [12.32, 11.47, 6.91, 0.86, 0.16]
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
            raw_cloud = np.fromfile(path, dtype=cloud_d_type)
            self.cloud_list.append(raw_cloud)

    def scan_to_proj_data(self, scan: LaserScan | SemLaserScan, labels):
        if self.gt:
            scan.set_label(labels)
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
        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
            :, None, None
        ]

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
        labels = self.cloud_list[index]["entity_id"].astype(np.uint32)
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


if __name__ == "__main__":
    sem_t4 = SemanticT4(True)
    # sem_t4.__getitem__(0)

    
