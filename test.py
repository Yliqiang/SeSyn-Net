import torch
from dataloader.data_pipeline import DataPipeline
from network.adjusted_stgcn import Adjusted_GCN
from torch.backends import cudnn
from torch.utils.data import DataLoader
import datetime as time
from matching import corresponding
from tqdm import tqdm
import config_test

opt = config_test.config()

torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)


def test():
    test_dataset = DataPipeline(opt.test_dataset)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=opt.test_batchsize,
                                 shuffle=False,
                                 num_workers=opt.workers,
                                 drop_last=True)

    length = len(test_dataset)

    checkpoint = torch.load(opt.checkpoint_path)

    model = Adjusted_GCN(opt.in_channels, opt.layout, opt.strategy, opt.edge_importance_weighting)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    res = []

    abs_error = 0
    rank_1 = 0
    rank_5 = 0
    rank_10 = 0

    relative_error_1 = 0
    relative_error_2 = 0

    rate_10 = 0
    rate_30 = 0
    rate_50 = 0

    number = 0

    for batch_i, batch_data in enumerate(tqdm(test_dataloader)):
        view1, view2 = batch_data

        data1, label1 = view1
        data2, label2 = view2

        data1 = data1.cuda()
        data2 = data2.cuda()

        label1 = label1.cuda()
        label2 = label2.cuda()

        label = label1 - label2

        output1 = model(data1)
        output2 = model(data2)

        frames = corresponding(output1, output2, label)

        res.append(frames)

        abs_error = abs_error + frames.item()

        if frames <= 1.0:
            rank_1 = rank_1 + 1
        if frames <= 5.0:
            rank_5 = rank_5 + 1
        if frames <= 10.0:
            rank_10 = rank_10 + 1

        GT = abs(label.item())

        relative_error_1 = relative_error_1 + frames.item()
        relative_error_2 = relative_error_2 + GT

        if GT == 0:
            number = number + 1
            continue

        relative_rate = frames / GT

        if relative_rate <= 0.1:
            rate_10 = rate_10 + 1
        if relative_rate <= 0.3:
            rate_30 = rate_30 + 1
        if relative_rate <= 0.5:
            rate_50 = rate_50 + 1

    res.sort()

    abs_error = abs_error / length
    rank_1 = rank_1 / length
    rank_5 = rank_5 / length
    rank_10 = rank_10 / length

    relative_error = relative_error_1 / relative_error_2
    rate_10 = rate_10 / (length - number)
    rate_30 = rate_30 / (length - number)
    rate_50 = rate_50 / (length - number)

    print('Abs_error:', abs_error, 'Rank_1:', rank_1, 'Rank_5:', rank_5, 'Rank_10:', rank_10)
    print('Relative_error:', relative_error, 'Rate_10:', rate_10, 'Rate_30:', rate_30, 'Rate_50:', rate_50)

    return abs_error


if __name__ == '__main__':
    start = time.datetime.now()

    accuracy = test()
    print('Accuracy', accuracy)

    end = time.datetime.now()
    print('Spent time: ', end - start)
