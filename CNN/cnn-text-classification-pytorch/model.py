import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, channel_dim, input_dim, embed_dim, output_dim, kernel_sizes, is_static=False, dropout_rate=0.5):
        super(CNN_Text, self).__init__()

        self.embed = nn.Embedding(input_dim, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, channel_dim, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(kernel_sizes) * channel_dim, output_dim)

        if is_static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        """

        Args:
            x: shape: (batch_size, seq_len)

        Returns:

        """
        # (batch_size, seq_len) --> (batch_size, input_dim, embed_dim)
        x = self.embed(x)

        # (batch_size, input_dim, embed_dim) --> (batch_size, 1, input_dim, embed_dim)
        x = x.unsqueeze(1)

        # [(batch_size, kernel_num, W), ...] * len(kernel_sizes)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # [(batch_size, kernel_num), ...] * len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # [(batch_size, kernel_num), ...] * len(kernel_sizes) --> (batch_size, len(kernel_sizes) * kernel_num)
        x = torch.cat(x, 1)

        # (batch_size, len(kernel_sizes) * kernel_num)
        x = self.dropout(x)

        # (batch_size, output_dim)
        logit = self.fc1(x)
        return logit

if __name__=="__main__":
    torch.manual_seed(123)
    x = torch.randint(10, [32, 23])
    model = CNN_Text(30, 20, 10, 2, [4, 3, 2])
    y = model(x)
    print(y.shape)
