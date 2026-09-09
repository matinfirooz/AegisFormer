from __future__ import annotations

import torch
import torch.nn.functional as F


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-4)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
