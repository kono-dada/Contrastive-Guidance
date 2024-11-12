from dataclasses import dataclass

@dataclass
class Config:
    in_channels: int = 1
    out_channels: int = 10
    hidden_channels: int =16
    beta_min: float = 0.0001
    beta_max: float = 0.02
    n_steps = 1000