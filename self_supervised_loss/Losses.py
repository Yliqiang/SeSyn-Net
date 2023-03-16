import torch
import torch.nn as nn
from torch.nn import functional as Fun


def get_similarity(view1, view2):
    norm1 = torch.sum(torch.square(view1), dim=1)
    norm1 = norm1.reshape(-1, 1)
    norm2 = torch.sum(torch.square(view2), dim=1)
    norm2 = norm2.reshape(1, -1)

    similarity = norm1 + norm2 - 2.0 * torch.matmul(view1, view2.transpose(1, 0))
    similarity = -1.0 * torch.maximum(similarity, torch.zeros(1).cuda())

    return similarity


def deviation(sim_21, label):
    softmax_sim_21 = Fun.softmax(sim_21, dim=1)

    back = torch.matmul(softmax_sim_21, label.float())

    error = abs(back - label).float()

    result = torch.mean(error)

    return result


def interval(sim_21, label):
    softmax_sim_21 = Fun.softmax(sim_21, dim=1)

    back = torch.matmul(softmax_sim_21, label.float())

    front = back[:-1]
    behind = back[1:]

    differ = behind - front - 1

    error = abs(differ)

    result = torch.mean(error)

    return result


def synchronization_sequences(view1, view2):
    length1 = view1.size(0)
    length2 = view2.size(0)

    sim_12 = get_similarity(view1, view2)
    softmaxed_sim_12 = Fun.softmax(sim_12, dim=1)

    nn_embs = torch.matmul(softmaxed_sim_12, view2)

    sim_21 = get_similarity(nn_embs, view1)

    labels_21 = torch.arange(length1)
    labels_21 = labels_21.cuda()

    labels_12 = torch.arange(length2)
    labels_12 = labels_12.cuda()

    return sim_21, labels_21, softmaxed_sim_12, labels_12


def identity(logits, labels):
    return Fun.cross_entropy(logits, labels)


class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

    def forward(self, view1, view2):
        sim_21, labels_21, softmaxed_sim_12, labels_12 = synchronization_sequences(view2[0], view1[0])

        identity = identity(sim_21, labels_21)

        deviation = deviation(sim_21, labels_21)

        interval = interval(sim_21, labels_21)

        # loss = identity + 0.5 * deviation + 0.035 * interval  # ntu
        loss = identity + 0.7 * deviation + 0.015 * interval  # cmu

        return loss
