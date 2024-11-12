from dataclasses import dataclass

@dataclass
class Config:
    in_channels: int = 1
    out_channels: int = 1
    hidden_dim_list: tuple = (32, 64)
    kernel_size: int = 3
    beta_min: float = 0.0001
    beta_max: float = 0.02
    n_steps = 1000