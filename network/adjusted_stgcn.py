import torch
import torch.nn as nn
from network.gconv_origin import ConvTemporalGraphical
from network.graph import Graph
from network.graph import Channel
from network.graph import Layers
import random

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
random.seed(1)


def iden(x):
    return x


class Adjusted_GCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 layout,
                 strategy,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        self.layout = layout
        self.graph = Graph(layout, strategy)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),

            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),

            st_gcn_block(64, 128, kernel_size, 1, **kwargs),

            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),

            st_gcn_block(128, 128, kernel_size, 1, **kwargs),

            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
        ))

        self.channel = Channel(gate_channels=17, reduction_ratio=2)

        self.frame_num = 2

        self.mlp = Layers(in_channels=self.frame_num, out_channels=1)

        self.ith = [i for i in range(len(self.st_gcn_networks))]

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        x1 = x[:, :, :, :, 0]

        if self.layout == '5':
            index = [0, 5, 6, 11, 12]
            index = torch.LongTensor(index).cuda()
            x1 = torch.index_select(x1, dim=3, index=index)

        if self.layout == '9':
            index = [0, 5, 6, 9, 10, 11, 12, 15, 16]
            index = torch.LongTensor(index).cuda()
            x1 = torch.index_select(x1, dim=3, index=index)

        if self.layout == '13':
            index = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            index = torch.LongTensor(index).cuda()
            x1 = torch.index_select(x1, dim=3, index=index)

        batch, channel, length, key = x1.size()

        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = x1.view(batch, key * channel, length)
        x1 = self.data_bn(x1)
        x1 = x1.view(batch, key, channel, length)
        x1 = x1.permute(0, 2, 3, 1).contiguous()

        for i, gcn, importance in zip(self.ith, self.st_gcn_networks, self.edge_importance):
            if i == 0:
                x1, _ = gcn(x1, self.A * importance)
            else:
                x1, _ = gcn(x1, self.A * importance)

        batch_size, dim1, seq_len, dim3 = x1.size()
        x1 = x1.permute(0, 2, 3, 1).contiguous()

        x1 = x1.view(batch_size, seq_len, -1)

        return x1


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


if __name__ == '__main__':
    model = ST_GCN_18(3, 'coco', 'spatial', True)
    print(model)
