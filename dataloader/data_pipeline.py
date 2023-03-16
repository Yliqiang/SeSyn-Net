import torch
from dataloader.loader import SkeletonLoader
from dataloader.skeketon_process import normalize_by_resolution, mask_by_visibility, construct_asynchronous, transpose, to_tuple
import config_test

opt = config_test.config()


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self, data_source):
        self.data_source = SkeletonLoader(data_source, opt.num_track, opt.num_keypoints)
        self.num_sample = len(self.data_source)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        two_view_data = self.data_source[index]

        data = two_view_data[0]
        data1 = two_view_data[1]

        if opt.mod == 'train':
            data = construct_asynchronous(data)
            data = normalize_by_resolution(data)
            data = mask_by_visibility(data)
            data = transpose(data, order=[0, 2, 1, 3])
            data = to_tuple(data)

            # *********************************************

            data1 = construct_asynchronous(data1)
            data1 = normalize_by_resolution(data1)
            data1 = mask_by_visibility(data1)
            data1 = transpose(data1, order=[0, 2, 1, 3])
            data1 = to_tuple(data1)

        return data, data1


if __name__ == '__main__':
    pass

