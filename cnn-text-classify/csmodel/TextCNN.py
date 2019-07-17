import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        in_channels = 1
        out_channels = args.kernel_num
        kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, embed_dim)
            )
                for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channels, class_num)

    def forward(self, x):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
