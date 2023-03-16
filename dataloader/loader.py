import os
import numpy as np
import json
import torch
import config_test

opt = config_test.config()


class SkeletonLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_track=1, num_keypoints=-1):

        self.data_dir = data_dir
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.files = [
                         os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                     ]

        self.view1 = []
        self.view2 = []

        if opt.dataset_name == 'NTU':
            for file in self.files:
                if 'C001' in file:
                    self.view1.append(file)
                if 'C002' in file:
                    self.view2.append(file)

        if opt.dataset_name == 'CMU':
            for file in self.files:
                if 'C11' in file:
                    self.view1.append(file)
                if 'C21' in file:
                    self.view2.append(file)

        self.view1.sort()
        self.view2.sort()

    def __len__(self):
        assert len(self.view1) == len(self.view2)
        return len(self.view1)

    def __getitem__(self, index):
        with open(self.view1[index]) as f:
            data = json.load(f)

        info = data['info']
        annotations = data['annotations']

        num_frame = info['num_frame']
        if num_frame > 1000:
            num_frame = int(len(data['annotations']) / 5)

        num_keypoints = info['num_keypoints']
        channel = info['keypoint_channels']
        num_channel = len(channel)

        # get data
        data['data'] = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']

            if person_id < self.num_track and frame_index < num_frame:
                data['data'][:, :, frame_index, person_id] = np.array(a['keypoints']).transpose()

        with open(self.view2[index]) as f1:
            data1 = json.load(f1)

        info1 = data1['info']
        annotations1 = data1['annotations']

        num_frame1 = info1['num_frame']
        if num_frame1 > 1000:
            num_frame1 = int(len(data1['annotations']) / 5)

        num_keypoints1 = info1['num_keypoints']
        channel1 = info1['keypoint_channels']
        num_channel1 = len(channel1)

        # get data
        data1['data'] = np.zeros(
            (num_channel1, num_keypoints1, num_frame1, self.num_track),
            dtype=np.float32)

        for a1 in annotations1:
            person_id1 = a1['id'] if a1['person_id'] is None else a1['person_id']
            frame_index1 = a1['frame_index']

            if person_id1 < self.num_track and frame_index1 < num_frame1:
                data1['data'][:, :, frame_index1, person_id1] = np.array(a1['keypoints']).transpose()

        return data, data1


