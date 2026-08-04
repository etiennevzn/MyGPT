import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if(isinstance(module, nn.ReLU)):
                    frac = round(((x == 0).all(dim = 0)).float().mean().item(), 4)
                    dead_fractions.append(frac)

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if len(dead_fractions) == 0 : return "healthy"

        max_frac = max(dead_fractions)
        if max_frac > 0.5 : return "use_leaky_relu"
        
        if dead_fractions[0] > 0.3 : return "reinitialize"
        
        increasing = True
        for i in range(1, len(dead_fractions)):
            if(dead_fractions[i] <= dead_fractions[i-1]):
                increasing = False
                break
        
        if(increasing and dead_fractions[-1] > 0.1) : return "reduce_learning_rate"

        if(max_frac < 0.1) : return "healthy"

        return "healthy"

