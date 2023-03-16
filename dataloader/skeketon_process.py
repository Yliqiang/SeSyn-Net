import random
import numpy as np

random.seed(1)
np.random.seed(1)


def normalize_by_resolution(data):
    resolution = data['info']['resolution']
    channel = data['info']['keypoint_channels']
    np_array = data['data']

    for i, c in enumerate(channel):
        if c == 'x':
            np_array[i] = np_array[i] / resolution[0] - 0.5
        if c == 'y':
            np_array[i] = np_array[i] / resolution[1] - 0.5

    data['data'] = np_array
    return data


def mask_by_visibility(data):
    channel = data['info']['keypoint_channels']
    np_array = data['data']

    for i, c in enumerate(channel):
        if c == 'score' or c == 'visibility':
            mask = (np_array[i] == 0)
            for j in range(len(channel)):
                if c != j:
                    np_array[j][mask] = 0

    data['data'] = np_array
    return data


def transpose(data, order, key='data'):
    data[key] = data[key].transpose(order)
    return data


def to_tuple(data, keys=None):
    if keys is None:
        keys = ['data', 'category_id']
    return tuple([data[k] for k in keys])


def construct_asynchronous(data):

    shift = data['category_id']

    data['data'] = data['data'][:, :, shift:, :]

    return data




