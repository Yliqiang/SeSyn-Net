import math
import torch
from torch.nn import functional as Fun
import numpy as np
import random

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
random.seed(1)


def get_similarity(view1, view2):
    norm1 = torch.sum(torch.square(view1), dim=1)
    norm1 = norm1.reshape(-1, 1)
    norm2 = torch.sum(torch.square(view2), dim=1)
    norm2 = norm2.reshape(1, -1)
    similarity = norm1 + norm2 - 2.0 * torch.matmul(view1, view2.transpose(1, 0))
    similarity = -1.0 * torch.maximum(similarity, torch.zeros(1).cuda())

    return similarity


def decision_offset(view1, view2, label):
    sim_12 = get_similarity(view1, view2)

    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    ground = (torch.tensor([i * 1.0 for i in range(view1.size(0))]).cuda()).reshape(-1, 1)

    predict = softmaxed_sim_12.argmax(dim=1)

    length1 = ground.size(0)

    frames = []

    for i in range(length1):
        p = predict[i].item()
        g = ground[i][0].item()

        frame_error = (p - g)
        frames.append(frame_error)

    median_frames = np.median(frames)

    num_frames = math.floor(median_frames)

    result = abs(num_frames - label)

    return result


def corresponding(view1, view2, label):
    result = decision_offset(view1[0], view2[0], label)

    return result


if __name__ == '__main__':
    output1 = torch.randn((1, 40, 2176)).cuda()
    output2 = torch.randn((1, 50, 2176)).cuda()

    label = torch.ones(1).cuda() * 10

    frames1, frames2 = corresponding(output1, output2, label)
    print(frames1, frames2.item())
