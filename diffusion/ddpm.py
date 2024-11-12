import torch
from torch import nn
from .unet import UNet
from .config import Config


class DiffusionPipeline:
    def __init__(self, device='cuda', config: Config = Config()):
        self.device = device
        self.unet = UNet(config.in_channels, config.out_channels, config.hidden_dim_list).to(device)
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(config.beta_min, config.beta_max, config.n_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = config.n_steps
        self.sigma2 = self.beta
        
    def predict_eps(self, x):  # c should start from 0
        batchsize = x.shape[0]
        t = torch.randint(self.n_steps, size=(batchsize,))
        alphas = self.alpha_bar[t].view(-1, 1, 1, 1)
        t = t.to(self.device)
        eps = torch.randn_like(x).to(self.device)
        zt = torch.sqrt(alphas) * x + torch.sqrt(1. - alphas) * eps
        predicted_noise = self.unet(zt, t)
        loss = nn.functional.mse_loss(predicted_noise, eps)
        return loss
    
    def from_pretrained(self, model_path):
        self.unet.load_state_dict(torch.load(model_path))
    
    @torch.no_grad()
    def predict(self, xt, t):
        size = xt.shape[0]
        x_hat = xt / (1 - self.beta[t]) ** 0.5 - self.beta[t] / ((1-self.alpha_bar[t]) ** 0.5 * (1-self.beta[t]) ** 0.5) * \
            self.unet(xt, torch.full((size,), t, dtype=torch.long).to(self.device))
        return x_hat
    
    @torch.no_grad()
    def ddpm_sample(self, n_samples, h=28, w=28):
        z = torch.randn((n_samples, 1, h, w)).to(self.device)
        size = z.shape[0]
        for t in reversed(range(self.n_steps)):
            z_hat = z / (1 - self.beta[t]) ** 0.5 - self.beta[t] / ((1-self.alpha_bar[t]) ** 0.5 * (1-self.beta[t]) ** 0.5) * \
                self.unet(z, torch.full((size,), t, dtype=torch.long).to(self.device))
            if t > 0:
                eps = torch.randn_like(z)
                z = z_hat + self.sigma2[t] ** 0.5 * eps
        return z_hat
    
    def cond_sample(self, n_samples, condition, contrastive_model: nn.Module, s, h=28, w=28):
        contrastive_model.eval()
        z = torch.randn((n_samples, 1, h, w), requires_grad=True).to(self.device)
        size = z.shape[0]
        for t in reversed(range(self.n_steps)):
            z_hat = self.predict(z, t)
            if t > 0:
                eps = torch.randn_like(z)
                z = z_hat + self.sigma2[t] ** 0.5 * eps
                z = z.requires_grad_(True)
                z_feature = contrastive_model(z, t)  # (n_samples, hidden_dim)
                with torch.no_grad():
                    condition_feature = contrastive_model(condition)  # (1, hidden_dim)
                similarity = torch.mm(z_feature, condition_feature.T)   # (n_samples, 1)
                # compute the grad to z
                grads = torch.zeros(size, 1, h, w).to(self.device)
                for i in range(size):
                    grad = torch.autograd.grad(similarity[i], z, retain_graph=True)[0][i]
                    grads[i] = grad
                z = z + s * grads
        return z_hat
    
def ddpm_from_config(device, config):
    return DiffusionPipeline(
        device=device,
        n_conditions=config['n_conditions'],
        features=config['features'],
        n_steps=config['n_steps'],
    )