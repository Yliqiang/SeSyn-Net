import torch
import os
from dataloader.data_pipeline import DataPipeline
from network.adjusted_stgcn import Adjusted_GCN
from self_supervised_loss.Losses import Losses
from torch.backends import cudnn
from torch.utils.data import DataLoader
import datetime as time
import config_test

opt = config_test.config()

torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)


def save_log(file, epoch_i, epoch_loss):
    log_line = 'Training epoch [' + str(epoch_i).zfill(3) + '], loss = ' + str(epoch_loss) + '\n'
    file.write(log_line)
    file.flush()


def save_model(epoch_i, epoch_loss, optimizer, model):
    pth_name = str(epoch_i).zfill(3) + '.pth'
    checkpoint_path = os.path.join(opt.work_dir, 'train', pth_name)
    torch.save(
        {
            'epoch': epoch_i,
            'loss': epoch_loss,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict()
        },
        checkpoint_path
    )


def train():
    log_name = os.path.join(opt.work_dir, 'train', 'log.txt')
    file = open(log_name, 'w')

    train_dataset = DataPipeline(opt.train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=opt.train_batchsize,
                                  shuffle=True,
                                  num_workers=opt.workers,
                                  drop_last=True,
                                  pin_memory=True)

    model = Adjusted_GCN(opt.in_channels, opt.layout, opt.strategy, opt.edge_importance_weighting)
    model.apply(weights_init)
    model = model.cuda()
    model.train()

    log_var_a = torch.zeros((1,)).cuda()
    log_var_b = torch.zeros((1,)).cuda()
    log_var_c = torch.zeros((1,)).cuda()
    log_var_d = torch.zeros((1,)).cuda()

    log_var_a.requires_grad = True
    log_var_b.requires_grad = True
    log_var_c.requires_grad = True
    log_var_d.requires_grad = True

    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d])

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.0001)

    self_loss = Losses()

    start = time.datetime.now()

    loss_list = []

    for epoch_i in range(opt.max_epochs):
        epoch_loss = 0

        for batch_i, batch_data in enumerate(train_dataloader):
            view1, view2 = batch_data

            data1, label1 = view1
            data2, label2 = view2

            data1 = data1.cuda()
            data2 = data2.cuda()

            label1 = label1.cuda()
            label2 = label2.cuda()

            output1 = model(data1)
            output2 = model(data2)

            loss = self_loss(output1, output2)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Training epoch {0}, loss = {1}'.format(epoch_i, epoch_loss))

        loss_list.append(epoch_loss)

        save_log(file, epoch_i, epoch_loss)

        if epoch_i % opt.save_fre == 0:
            save_model(epoch_i, epoch_loss, optimizer, model)

    end = time.datetime.now()

    cost = end - start
    print('Spend time：', cost)

    file.close()


if __name__ == '__main__':
    train()
