import torch
import torch.nn as nn


class TemplateModel(nn.Module):
    """LeNet-5 classification model.""" 
    def __init__(
            self, 
            in_channels: int,
            hidden_dim: int,
            out_dim: int
    ) -> None:
        super().__init__()

        # Basically a LeNet-5 style architecture:
        # https://en.wikipedia.org/wiki/LeNet
        # http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
        self.conv_net_2d = nn.Sequential(
            # [B, in_channels, 28, 28] -> [B, 6, 24, 24]
            nn.Conv2d(in_channels, 6, 5, 1, 0),
            nn.ReLU(),
            
            # [B, 6, 24, 24] -> [B, 6, 12, 12]
            nn.MaxPool2d(2, 2),
        
            # [B, 6, 12, 12] -> [B, 16, 8, 8]
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            
            # [B, 16, 8, 8] -> [B, 16, 4, 4]
            nn.MaxPool2d(2, 2),
        
            # [B, 16, 4, 4] -> [B, 120, 1, 1]
            nn.Conv2d(16, 120, 4, 1, 0),
            nn.ReLU(),
            
            # [B, 120, 1, 1] -> [B, 120 * 3 * 3] = [B, 120]
            nn.Flatten(start_dim=1)
        )
        
        self.cls_mlp = nn.Sequential(
            # [B, 120] -> [B, hidden_dim]
            nn.Linear(120, hidden_dim),
            nn.ReLU(),

            # [B, hidden_dim] -> [B, out_dim]
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_latent = self.conv_net_2d(x)  # [B, hidden_dim]
        return self.cls_mlp(x_latent)   # [B, out_dim]
