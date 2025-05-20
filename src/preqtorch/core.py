import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import os, random

class ReplayStreams:
    def __init__(self, n_streams, reset_prob_fn=None):
        """
        Manages replay streams for online learning.

        Args:
            n_streams: Number of independent replay streams.
            reset_prob_fn: Function f(t) giving probability of reset at iteration t (default = 1/(t+2))
        """
        self.dataset = []
        self.n_streams = n_streams
        self.stream_indices = [0 for _ in range(n_streams)]
        self.reset_prob_fn = reset_prob_fn or (lambda t: 1 / (t + 2))
        self.t = 0

    def update(self, batch):
        self.dataset.append(batch)
    def sample(self):
        """
        Returns n_streams batches for the given iteration t.
        May reset stream index with probability defined by reset_prob_fn.

        Returns:
            List[Tuple[batch_idx, batch_data]]
        """
        sampled = []
        for i in range(self.n_streams):
            index = self.stream_indices[i]
            batch = self.dataset[index]
            sampled.append((index, batch))

            if random.random() < self.reset_prob_fn(self.t):
                self.stream_indices[i] = 0
            else:
                self.stream_indices[i] = min(self.stream_indices[i] + 1, len(self.dataset) - 1)

        self.t += 1

        return sampled


class PrequentialEncoder:
    """
    Base class for prequential encoding methods.

    This class defines the common interface and functionality for all prequential encoders.
    Subclasses should implement the specific encoding methods.

    Note: This class only works with datasets that return batches in the following format:
    - Either tuples of tensors
    - Or tuples of tuples including tensors
    """

    def __init__(self, model_class, loss_fn=None, device=None, optimizer_fn=None):
        """
        Initialize the encoder.

        Args:
            model_class: A PyTorch model class that will be instantiated for encoding
            loss_fn: Encoding function (if None, cross_entropy will be used)
                     This function should return per-sample code lengths
            device: Device to run the model on (if None, will use cuda if available, else cpu)
            optimizer_fn: Function to create optimizer (if None, Adam will be used)
        """
        self.model_class = model_class
        self.encoding_fn = loss_fn  # Renamed for clarity
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer_fn = optimizer_fn

    def to(self, device):
        """
        Move the encoder to the specified device.

        Args:
            device: The device to move to (e.g., 'cuda', 'cpu', torch.device)

        Returns:
            self: Returns self for method chaining
        """
        self.device = device
        return self

    def _move_to_device(self, obj):
        """
        Move an object to the device.

        If the object is a tensor, move it to the device.
        If the object is a tuple, recursively move each element to the device if it's a tensor.
        Otherwise, leave the object as is.

        Args:
            obj: The object to move to the device

        Returns:
            The object moved to the device
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(item) for item in obj)
        else:
            return obj

    def _move_to_cpu(self, obj):
        """
        Move an object to CPU if the current device is not CPU.

        If the object is a tensor and the current device is not CPU, move it to CPU.
        If the object is a tuple, recursively move each element to CPU if it's a tensor.
        Otherwise, leave the object as is.

        Args:
            obj: The object to move to CPU

        Returns:
            The object moved to CPU if needed
        """
        if self.device == 'cpu' or self.device == torch.device('cpu'):
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, tuple):
            return tuple(self._move_to_cpu(item) for item in obj)
        else:
            return obj

    def _get_default_encoding_fn(self):
        """
        Returns the default encoding function if none is provided.
        The encoding function returns per-sample code lengths.
        """
        def encoding_fn(outputs, targets, mask):
            try:
                return torch.nn.functional.cross_entropy(outputs[mask], targets[mask], reduction='none')
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    raise RuntimeError(
                        "Device mismatch error. Please ensure your model handles device placement "
                        "for inputs and outputs in its forward method."
                    ) from e
                raise
        return encoding_fn

    def _get_optimizer(self, model, learning_rate):
        """
        Returns the optimizer for the model.
        """
        if self.optimizer_fn is None:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            return self.optimizer_fn(model.parameters(), lr=learning_rate)

    def _sample_model_class(self):
        """
        Samples a model from self.model_class.

        If the model class has an 'initialize' function, it calls it and returns a deepcopy
        of the model returned by initialize. Otherwise, it initializes a new model instance
        using xavier uniform initialization and returns a deepcopy.

        Returns:
            A deepcopy of the initialized model.
        """
        # Check if model_class has an initialize function
        if hasattr(self.model_class, 'initialize') and callable(getattr(self.model_class, 'initialize')):
            # Call initialize and return a deepcopy of the result
            model = self.model_class.initialize()
            return deepcopy(model)
        else:
            # Initialize a new model instance
            model = self.model_class()

            # Apply xavier uniform initialization to all parameters
            for param in model.parameters():
                if param.dim() > 1:  # Only apply to weight matrices, not bias vectors
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in str(param): # bias vectors init zero
                    torch.nn.init.zeros_(param)
                else: # else uniform initialization, even on [-1, 1]
                    torch.nn.init.uniform_(param, a = -1.0, b = 1.0)

            return deepcopy(model)

    def encode(self, *args, **kwargs):
        """
        Encode the data using the prequential coding method.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")