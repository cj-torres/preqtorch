import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from copy import deepcopy
import os, random


class ModelClass:
    """
    Creates a wrapper for torch.nn.Module to allow sampling new models.
    This class provides functionality to initialize and manage PyTorch models
    with consistent initialization strategies. It ensures models are properly
    initialized with Xavier/Uniform weights and handles device placement.
    """
    def __init__(self, model: type, device: str, kwargs: dict = None, init_func=None):
        """
        Initialize the ModelClass wrapper.

        Args:
            model (type): A torch.nn.Module subclass (not instance) to be wrapped
            device (str): The device to place models on ('cpu' or 'cuda')
            kwargs (dict): Optional keyword arguments to pass to the model constructor
            init_func (callable, optional): Custom initialization function for model parameters.
                                           If None, default initialization is used.
        """
        assert issubclass(model, torch.nn.Module), "model must be a subclass of torch.nn.Module"
        self.model = model
        self.device = device
        self.init_func = init_func
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}

    def __contains__(self, model_instance: torch.nn.Module):
        """
        Args:
            model_instance: an example model

        Returns:
            Whether the model is a member of the Model Class
        """
        return isinstance(model_instance, self.model)

    def initialize(self):
        """
        Creates and initializes a new instance of the model.
        If a custom initialization function was provided, it will be used.
        Otherwise, the default initialization strategy will be applied.

        Returns:
            torch.nn.Module: A newly initialized model instance on the specified device
        """
        new_model = self.model(**self.kwargs)
        new_model.to(self.device)

        if self.init_func is not None:
            return self.init_func(new_model)
        else:
            return self._initialize_model(new_model)

    def to(self, device):
        """
        Updates the target device for future model instantiations.

        Args:
            device (str): The new target device ('cpu' or 'cuda')
        """
        self.device = device

    @staticmethod
    def _initialize_model(model):
        """
        Initializes model parameters using a specific strategy:
        - Xavier uniform initialization for weight matrices
        - Zero initialization for bias vectors
        - Uniform [-1, 1] initialization for other parameters

        Args:
            model (torch.nn.Module): The model to initialize

        Returns:
            torch.nn.Module: The initialized model
        """
        # Apply xavier uniform initialization to all parameters
        for param in model.parameters():
            if param.dim() > 1:  # Only apply to weight matrices, not bias vectors
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in str(param):  # bias vectors init zero
                torch.nn.init.zeros_(param)
            else:  # else uniform initialization, even on [-1, 1]
                torch.nn.init.uniform_(param, a=-1.0, b=1.0)
        return model
