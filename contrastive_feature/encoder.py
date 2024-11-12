import math
import torch
from torch import nn
from .config import Config

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ContrastiveEncoder(nn.Module):
    def __init__(self, device, config: Config = Config()):
        super(ContrastiveEncoder, self).__init__()
        self.device = device
        in_channels = config.in_channels
        out_channels = config.out_channels
        hidden_channels = config.hidden_channels
        self.conv1 = nn.Conv2d(in_channels,  hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels,  hidden_channels*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.time_embedding = TimeEmbedding(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels*2,  out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels,  out_channels, kernel_size=1)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = Swish()
        
        self.beta = torch.linspace(config.beta_min, config.beta_max, config.n_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = config.n_steps
        self.sigma2 = self.beta
        
    def forward(self, x, t=None):
        if t is None:
            t = 0
        if type(t) == int:
            t = torch.full((x.shape[0],), t, dtype=torch.long).to(self.device)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x += self.time_embedding(t)[:, :, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv4(x) + x
        # x = self.pool(x)
        # x = self.batch_norm(x)
        # sum up all the channels
        x = x.sum(dim=(2, 3))
        # normalize the feature
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def add_noise(self, x, t=None):
        batchsize = x.shape[0]
        if t is None:
            t = torch.randint(self.n_steps, size=(batchsize,))
        alphas = self.alpha_bar[t].view(-1, 1, 1, 1)
        t = t.to(self.device)
        eps = torch.randn_like(x).to(self.device)
        zt = torch.sqrt(alphas) * x + torch.sqrt(1. - alphas) * eps
        return zt, t
    
    def training_features(self, x):
        zt, t = self.add_noise(x)
        return self.forward(zt, t)
    
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