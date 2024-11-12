import math
import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ResDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResDownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels,  out_channels, kernel_size=kernel_size, padding=1)
        self.time_embedding = TimeEmbedding(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = Swish()
        
    def forward(self, x, t):
        x = self.conv1(x)
        x = self.activation(x)
        x += self.time_embedding(t)[:, :, None, None]
        x = self.conv2(x) + x
        x = self.activation(x)
        x = self.pool(x)
        return x
    
class ResUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResUpSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2,  in_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels,  out_channels, kernel_size=kernel_size, padding=1)
        self.time_embedding = TimeEmbedding(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = Swish()
        
    def forward(self, x, skip_con_x, t):
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.activation(x)
        x += self.time_embedding(t)[:, :, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        return x
    
class Bottleneck(nn.Module):
    def __init__(self, input_channels, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=kernel_size, padding=1)
        self.time_embedding = TimeEmbedding(input_channels * 2)
        self.activation = Swish()
        
    def forward(self, x, t):
        _x = self.conv1(x)
        _x = self.activation(_x)
        _x += self.time_embedding(t)[:, :, None, None]
        _x = self.conv2(_x) + x
        _x = self.activation(_x)
        return _x
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_list, kernel_size=3):
        super(UNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim_list[0], kernel_size=kernel_size, padding=1) 
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.bottleneck = Bottleneck(hidden_dim_list[-1], kernel_size)
        
        in_channels = hidden_dim_list[0]
        for hidden_dim in hidden_dim_list[1:]:
            self.downs.append(ResDownSample(in_channels, hidden_dim, kernel_size))
            in_channels = hidden_dim
            
        hidden_dim_list = hidden_dim_list[::-1]
        for hidden_dim in hidden_dim_list[1:]:
            self.ups.append(ResUpSample(in_channels, hidden_dim, kernel_size))
            in_channels = hidden_dim
        
        self.conv_out = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, t):
        skips = []
        x = self.conv_in(x)
        _x = x
        for down in self.downs:
            x = down(x, t)
            skips.append(x)
        x = self.bottleneck(x, t)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t)
        x = self.conv_out(torch.cat([_x, x], axis=1))
        return x
        
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    This class is from labml.nn
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb