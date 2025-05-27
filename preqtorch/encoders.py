import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import os, random
from .utils import ModelClass
from .replay import ReplayStreams, ReplayBuffer, Replay, ReplayingDataLoader


class EncoderState:
    """
    Holds the state of the encoding process. Allows for pausing/resuming training.
    """
    def __init__(self, model, optim, beta, beta_optim, ema_params, trained_params):
        self.model = model
        self.optim = optim
        self.beta = beta
        self.beta_optim = beta_optim
        self.ema_params = ema_params
        self.trained_params = trained_params
        self.code_length = 0
        self.history = []

    def __repr__(self):
        return f"EncoderState(\ncode_length={self.code_length},\nhistory={self.history}\n)"

    def __str__(self):
        return self.__repr__()


class PrequentialEncoder:
    """
    Base class for prequential encoding methods.

    This class defines the common interface and functionality for all prequential encoders.
    Subclasses should implement the specific encoding methods.

    Note: This class only works with datasets that return batches in the following format:
    - Either tuples of tensors
    - Or tuples of tuples including tensors
    """

    def __init__(self, model_class: ModelClass, loss_fn=None, device=None, optimizer_fn=None):
        """
        Initialize the encoder.

        Args:
            model_class: A ModelClass object that will be used for model instantiation
            loss_fn: Encoding function (if None, cross_entropy will be used)
                     This function should return per-sample code lengths
            device: Device to run the model on (if None, will use cuda if available, else cpu)
            optimizer_fn: Function to create optimizer (if None, Adam will be used)
        """
        self.model_class = model_class
        self.encoding_fn = loss_fn  # Renamed for clarity
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Update the model_class device to match the encoder's device
        if hasattr(self.model_class, 'to'):
            self.model_class.to(self.device)
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
        # Also update the model_class device to match the encoder's device
        if hasattr(self.model_class, 'to'):
            self.model_class.to(device)
        return self

    def _move_to_device(self, obj):
        """
        Move an object to the device.

        If the object is a tensor, move it to the device.
        If the object is a tuple or list, recursively move each element to the device.
        If the object is a dictionary, recursively move each value to the device.
        Otherwise, leave the object as is.

        Args:
            obj: The object to move to the device

        Returns:
            The object moved to the device
        """
        if isinstance(obj, torch.Tensor):
            # Check if tensor is already on the target device
            if obj.device == self.device or obj.device == torch.device(self.device):
                return obj
            return obj.to(self.device)
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(item) for item in obj)
        elif isinstance(obj, list):
            return [self._move_to_device(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._move_to_device(value) for key, value in obj.items()}
        else:
            return obj

    def _move_to_cpu(self, obj):
        """
        Move an object to CPU if the current device is not CPU.

        If the object is a tensor and the current device is not CPU, move it to CPU.
        If the object is a tuple or list, recursively move each element to CPU.
        If the object is a dictionary, recursively move each value to CPU.
        Otherwise, leave the object as is.

        Args:
            obj: The object to move to CPU

        Returns:
            The object moved to CPU if needed
        """
        # If the encoder is already on CPU, return the object as is
        if self.device == 'cpu' or self.device == torch.device('cpu'):
            return obj

        if isinstance(obj, torch.Tensor):
            # Check if tensor is already on CPU
            if obj.device.type == 'cpu':
                return obj
            return obj.cpu()
        elif isinstance(obj, tuple):
            return tuple(self._move_to_cpu(item) for item in obj)
        elif isinstance(obj, list):
            return [self._move_to_cpu(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._move_to_cpu(value) for key, value in obj.items()}
        else:
            return obj

    def _get_default_encoding_fn(self):
        """
        Returns the default encoding function if none is provided.
        The encoding function returns per-sample code lengths.

        The encoding function takes outputs, targets, output_mask, and target_mask as parameters
        and applies the masks before computing the loss.
        """
        def encoding_fn(outputs, targets, output_mask, target_mask):
            return torch.nn.functional.cross_entropy(outputs[output_mask], targets[target_mask], reduction='none')/torch.log(torch.tensor(2.0, device=outputs.device))
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

        Uses the ModelClass.initialize() method to create and initialize a new model instance.
        Returns an initialized model.

        Returns:
            An initialized model.
        """
        # Use ModelClass.initialize() to create and initialize a new model instance
        model = self.model_class.initialize()
        return model

    def encode(self, *args, **kwargs):
        """
        Encode the data using the prequential coding method.

        This is a one-shot method that performs the following steps:
        1. Initialize the encoder with a dataset
        2. Step through batches
        3. Finalize to get the model and code length

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")


class BlockEncoder(PrequentialEncoder):
    """
    Prequential encoder using a staged block-wise learning approach.
    """
    def __init__(self, model_class: ModelClass, loss_fn=None, device=None, optimizer_fn=None):
        super().__init__(model_class, loss_fn, device, optimizer_fn)

    def encode(self, dataset, set_name, stop_points, batch_size, seed, 
               learning_rate=1e-4, epochs=50, patience=20, shuffle=True, 
               collate_fn=None, use_device_handling=True, num_samples=None):
        """
        One-shot method to encode the data using the block-wise prequential coding method.

        This method performs the following steps:
        1. Initialize the encoder with a dataset
        2. Step through batches
        3. Return the model and code length

        Args:
            dataset: The dataset to encode
            set_name: Name of the dataset (for logging)
            stop_points: List of points where to stop and evaluate
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            learning_rate: Learning rate for the optimizer
            epochs: Maximum number of epochs to train
            patience: Number of epochs to wait for improvement before early stopping
            shuffle: Whether to shuffle the data
            collate_fn: Function to collate samples into batches
            return_code_length_history: Whether to return the code length history
            use_device_handling: Whether to handle device placement in the model
            num_samples: Number of samples to use (if None, use all)

        Returns:
            If return_code_length_history is False:
                model: The trained model
                code_length: The code length of the encoded data
            If return_code_length_history is True:
                model: The trained model
                code_length: The code length of the encoded data
                code_length_history: The history of code lengths during training
        """
        # If num_samples is provided, create a subset of the dataset
        if num_samples is not None and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        # Initialize the encoder
        state, train_chunks, eval_chunks, batch_size, shuffle, collate_fn = self.initialize(
            dataset, stop_points, batch_size, learning_rate, seed, shuffle, collate_fn)

        encoding_fn = self.encoding_fn or self._get_default_encoding_fn()

        initial_weights = deepcopy(state.model.state_dict())

        for train_set, eval_set in zip(train_chunks, eval_chunks):
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
            eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Evaluate code length before training
            self.eval_code_length(state, eval_loader, encoding_fn)

            if train_set == train_chunks[-1]:
                break

            state.model.load_state_dict(deepcopy(initial_weights))
            self.train_until_patience(state, train_loader, encoding_fn, patience, epochs)

        print(f"Performance for {set_name}: Prequential code length: {state.code_length}")

        return state.model, state.code_length, state.history

    def calculate_code_length(self, model, batch, encoding_fn=None):
        encoding_fn = encoding_fn or self.encoding_fn or self._get_default_encoding_fn()
        inputs, target = batch[:2]
        inputs = self._move_to_device(inputs)
        target = self._move_to_device(target)

        # Handle different batch formats:
        # 1. input, target
        # 2. input, target, mask (shared mask for both input and target)
        # 3. input, target, output_mask, target_mask (separate masks for outputs and targets)

        if len(batch) <= 2 or batch[2] is None:
            # Case 1: No masks provided, create default masks
            output_mask = torch.ones_like(inputs, dtype=torch.bool, device=self.device)
            target_mask = torch.ones_like(target, dtype=torch.bool, device=self.device)
        elif len(batch) == 3:
            # Case 2: One mask provided, use it for both output and target
            shared_mask = self._move_to_device(batch[2])
            output_mask = shared_mask
            target_mask = shared_mask
        else:
            # Case 3: Two masks provided, use them separately
            output_mask = self._move_to_device(batch[2]) if batch[2] is not None else torch.ones_like(inputs, dtype=torch.bool, device=self.device)
            target_mask = self._move_to_device(batch[3]) if batch[3] is not None else torch.ones_like(target, dtype=torch.bool, device=self.device)

        # Generate model outputs
        outputs = model(inputs)

        # Pass outputs, targets, and masks to encoding_fn
        code_lengths = encoding_fn(outputs, target, output_mask, target_mask)
        return code_lengths, inputs, target, target_mask, output_mask

    def initialize(self, dataset, stop_points, batch_size, learning_rate, seed,
                   shuffle=True, collate_fn=None):
        torch.manual_seed(seed)
        random.seed(seed)

        model = self._sample_model_class()
        model.to(self.device)
        optim = self._get_optimizer(model, learning_rate)

        state = EncoderState(model=model, optim=optim, beta=None, beta_optim=None,
                             ema_params=None, trained_params=None)

        if stop_points[-1] != 1:
            stop_points.append(1)
        if stop_points[0] != 0:
            stop_points.insert(0, 0)

        chunk_sizes = [j - i for i, j in zip(stop_points[:-1], stop_points[1:])]
        chunks = torch.utils.data.random_split(dataset, chunk_sizes)
        train_chunks = [torch.utils.data.ConcatDataset(chunks[:i + 1]) for i in range(len(chunks))]

        return state, train_chunks, chunks, batch_size, shuffle, collate_fn

    def step(self, state, batch, encoding_fn=None):
        encoding_fn = encoding_fn or self.encoding_fn or self._get_default_encoding_fn()
        model = state.model
        optim = state.optim

        optim.zero_grad()
        code_lengths, inputs, target, target_mask, output_mask = self.calculate_code_length(model, batch, encoding_fn)
        loss = code_lengths.sum()
        loss.backward()
        optim.step()

        state.code_length += loss.item()
        state.history.append(loss.item())

    def eval_code_length(self, state, dataloader, encoding_fn=None):
        encoding_fn = encoding_fn or self.encoding_fn or self._get_default_encoding_fn()
        model = state.model
        model.eval()

        for batch in dataloader:
            code_lengths, inputs, target, target_mask, output_mask = self.calculate_code_length(model, batch, encoding_fn)
            state.code_length += code_lengths.sum().item()
            state.history.append(code_lengths.sum().item())

    def train_until_patience(self, state, train_dataloader, encoding_fn, patience, epochs):
        best_loss = float('inf')
        no_improvement = 0
        model = state.model
        optim = state.optim
        model.train()

        for epoch in range(epochs):
            for batch in train_dataloader:
                encoding_fn = encoding_fn or self._get_default_encoding_fn()
                optim.zero_grad()
                code_lengths, inputs, target, target_mask, output_mask = self.calculate_code_length(model, batch, encoding_fn)
                loss = code_lengths.sum()
                loss.backward()
                optim.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement > patience:
                    return



class MIREncoder(PrequentialEncoder):
    def __init__(self, model_class, loss_fn=None, device=None, optimizer_fn=None):
        super().__init__(model_class, loss_fn, device, optimizer_fn)

    def encode(self, dataset, set_name, n_replay_samples, learning_rate=1e-4, batch_size=32,
               seed=42, alpha=0.1, collate_fn=None, use_device_handling=True, use_beta=True,
               use_ema=True, shuffle=True, num_samples=None, replay_type="buffer"):
        """
        One-shot method to encode the data using the MIR prequential coding method.

        This method performs the following steps:
        1. Initialize the encoder with a dataset
        2. Step through batches
        3. Finalize to get the model and code length

        Args:
            dataset: The dataset to encode
            set_name: Name of the dataset (for logging)
            n_replay_samples: Number of replay streams or buffer size
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            alpha: EMA update rate
            collate_fn: Function to collate samples into batches
            use_device_handling: Whether to handle device placement in the model
            use_beta: Whether to use learnable temperature parameter
            use_ema: Whether to use exponential moving average
            shuffle: Whether to shuffle the data
            num_samples: Number of samples to use (if None, use all)
            replay_type: Type of replay to use ("buffer" or "streams")

        Returns:
            If return_code_length_history is False:
                model: The trained model
                code_length: The code length of the encoded data
                ema_params: The EMA parameters
                beta: The learnable temperature parameter
                replay_streams: The replay streams
            If return_code_length_history is True:
                model: The trained model
                code_length: The code length of the encoded data
                code_length_history: The history of code lengths during training
                ema_params: The EMA parameters
                beta: The learnable temperature parameter
                replay_streams: The replay streams
        """
        # If num_samples is provided, create a subset of the dataset
        if num_samples is not None and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        # Initialize the encoder
        state, replay_loader = self.initialize(
            dataset, batch_size, seed, n_replay_samples, replay_type,
            None, learning_rate, alpha, collate_fn, shuffle)

        # If not using beta or EMA, set them to None
        if not use_beta:
            state.beta = None
            state.beta_optim = None

        if not use_ema:
            state.ema_params = None

        # Process each batch
        for batch in replay_loader:
            self.step(batch, replay_loader, state, alpha)

        # Finalize and get results
        model, code_length, history, ema_params, beta = self.finalize(state)
        return model, code_length, history, ema_params, beta, replay_loader.replay

    def initialize(self, dataset, batch_size, seed, n_replay_samples, replay_type="buffer",
                   model=None, learning_rate=1e-4, alpha=0.1, collate_fn=None, shuffle=True):
        """
        Initializes model, replay loader, optimizer, and state tracking.
        """
        torch.manual_seed(seed)
        random.seed(seed)

        model = model or self._sample_model_class()
        model.to(self.device)

        optim = self._get_optimizer(model, learning_rate)
        beta = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        beta_optim = torch.optim.Adam([beta], lr=learning_rate)

        # Select replay type
        if replay_type == "streams":
            replay_impl = ReplayStreams(dataset, batch_size=batch_size, n_streams=n_replay_samples, collate_fn=collate_fn)
        elif replay_type == "buffer":
            replay_impl = ReplayBuffer(dataset, batch_size=batch_size, n_samples=n_replay_samples, collate_fn=collate_fn)
        else:
            raise ValueError("replay_type must be 'streams' or 'buffer'")

        replay_loader = ReplayingDataLoader(dataset, batch_size=batch_size, replay=replay_impl,
                                            collate_fn=collate_fn, shuffle=shuffle)

        ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        trained_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        state = EncoderState(model, optim, beta, beta_optim, ema_params, trained_params)
        return state, replay_loader

    def step(self, batch, replay_loader, state, alpha=0.1):
        encoding_fn = self.encoding_fn or self._get_default_encoding_fn()
        model = state.model
        beta = state.beta

        use_beta = beta is not None
        state.optim.zero_grad()
        if use_beta:
            state.beta_optim.zero_grad()

        # New batch forward pass
        code_lengths, inputs, target, target_mask, output_mask = self.calculate_code_length(
            model, batch, beta, state.ema_params, encoding_fn)
        loss = code_lengths.sum()
        state.code_length += loss.detach()
        state.history.append(loss.detach())
        loss.backward()
        if use_beta:
            state.beta_optim.step()
        state.optim.step()

        # Update EMA
        if state.ema_params is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    state.ema_params[name] = state.ema_params[name] * (1 - alpha) + param * alpha

        # Replay training
        for _, replay_batch in replay_loader.sample_replay():
            state.optim.zero_grad()
            code_lengths, _, _, _, _ = self.calculate_code_length(model, replay_batch, None, None, encoding_fn)
            loss = code_lengths.sum()
            loss.backward()
            state.optim.step()
            if state.ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        state.ema_params[name] = state.ema_params[name] * (1 - alpha) + param * alpha

    def calculate_code_length(self, model, batch, beta=None, ema_params=None, encoding_fn=None):
        """
        Calculates the code length for a single batch of data without updating the model.

        Args:
            model: PyTorch model.
            batch: Tuple containing inputs, targets, and optionally masks.
                  The batch can be in one of three formats:
                  1. (inputs, targets)
                  2. (inputs, targets, mask) - shared mask for both inputs and targets
                  3. (inputs, targets, output_mask, target_mask) - separate masks for outputs and targets
            beta: Optional learnable temperature scalar.
            ema_params: Optional EMA model parameters.
            encoding_fn: Optional override of the default encoding function.

        Returns:
            Tuple: (code_lengths, inputs, targets, target_mask, output_mask)
        """
        encoding_fn = encoding_fn or self.encoding_fn or self._get_default_encoding_fn()

        # Optionally swap model weights with EMA weights
        original_params = {}
        if ema_params is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    original_params[name] = param.clone().detach()
                    param.data.copy_(ema_params[name].data)

        inputs, target = batch[:2]
        inputs = self._move_to_device(inputs)
        target = self._move_to_device(target)

        # Handle different batch formats:
        # 1. input, target
        # 2. input, target, mask (shared mask for both input and target)
        # 3. input, target, output_mask, target_mask (separate masks for outputs and targets)

        if len(batch) <= 2 or batch[2] is None:
            # Case 1: No masks provided, create default masks
            output_mask = torch.ones_like(inputs, dtype=torch.bool, device=self.device)
            target_mask = torch.ones_like(target, dtype=torch.bool, device=self.device)
        elif len(batch) == 3:
            # Case 2: One mask provided, use it for both output and target
            shared_mask = self._move_to_device(batch[2])
            output_mask = shared_mask
            target_mask = shared_mask
        else:
            # Case 3: Two masks provided, use them separately
            output_mask = self._move_to_device(batch[2]) if batch[2] is not None else torch.ones_like(inputs, dtype=torch.bool, device=self.device)
            target_mask = self._move_to_device(batch[3]) if batch[3] is not None else torch.ones_like(target, dtype=torch.bool, device=self.device)

        try:
            # Generate model outputs
            outputs = model(inputs)
            if beta is not None:
                outputs = outputs * F.softplus(beta)

            # Pass outputs, targets, and masks to encoding_fn
            code_lengths = encoding_fn(outputs, target, output_mask, target_mask)

            # Restore original weights if EMA was used
            if ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.data.copy_(original_params[name].data)

            inputs = self._move_to_cpu(inputs)
            target = self._move_to_cpu(target)
            target_mask = self._move_to_cpu(target_mask)
            output_mask = self._move_to_cpu(output_mask)

            return code_lengths, inputs, target, target_mask, output_mask

        except Exception as e:
            # Restore original weights in case of failure
            if ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.data.copy_(original_params[name].data)
            raise e

    def finalize(self, state):
        """Returns final encoding stats after training."""
        if self.device != 'cpu':
            state.beta.data = state.beta.data.cpu()
            for name in state.ema_params:
                state.ema_params[name] = state.ema_params[name].cpu()

        return state.model, state.code_length, state.history, state.ema_params, state.beta
