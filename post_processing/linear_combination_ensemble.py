from copy import deepcopy
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class LinearCombinationEnsemble(nn.Module):
    """A wrapper over several models averaging their predictions with some weights.
    """
    def __init__(self, models, weight_logits=None):
        """Initializes LinearCombinationEnsemble.
        Parameters
        ----------
        models : List[torch.nn.Module]
            The model that the aggregator wraps.
        weight_logits
            determines the weights with which model predictions will be averaged.
             weights = softmax(weight_logits)
        """
        assert (len(models) > 1)

        super().__init__()

        self.models = [deepcopy(model) for model in models]
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.weight_logits = weight_logits if weight_logits else np.zeros(len(models))
        self.weight_logits = torch.tensor(self.weight_logits, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        weighted_predictions = []

        weights = F.softmax(self.weight_logits, dim=0)
        for model, weight in zip(self.models, weights):
            weighted_predictions.append(weight * model(x))

        return torch.stack(weighted_predictions).sum(axis=0)
