import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import os, random

from .core import ReplayStreams


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

class BlockEncoder(PrequentialEncoder):
    """
    Prequential encoder using block-based approach.
    """

    def __init__(self, model_class, loss_fn=None, device=None, optimizer_fn=None):
        """
        Initialize the block encoder.

        Args:
            model_class: A PyTorch model class that will be instantiated for encoding
            loss_fn: Encoding function (if None, cross_entropy will be used)
                     This function should return per-sample code lengths
            device: Device to run the model on (if None, will use cuda if available, else cpu)
            optimizer_fn: Function to create optimizer (if None, Adam will be used)
        """
        super().__init__(model_class, loss_fn, device, optimizer_fn)

    def calculate_code_length(self, model, batch, encoding_fn=None, with_grad=False):
        """
        Calculates the code length for a single batch of data without updating the model.

        Args:
            model: PyTorch model.
            batch: Tuple containing inputs and target.
            encoding_fn: Encoding function. If None, will use the encoder's encoding function.
                         This function should return per-sample code lengths.
            with_grad: Whether to calculate gradients. If False, will use torch.no_grad().

        Returns:
            Tuple of (code_length, inputs, target, target_mask)
        """
        # Set encoding function if not provided
        if encoding_fn is None:
            encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        # Handle different dataset formats
        inputs, target = batch[:2]

        # Move inputs and target to device
        inputs = self._move_to_device(inputs)
        target = self._move_to_device(target)

        # If there's a mask provided as 3rd element, use it
        if len(batch) > 2 and batch[2] is not None:
            target_mask = self._move_to_device(batch[2])
        else:
            # Default mask calculation
            target_mask = torch.ones_like(target, dtype=torch.bool, device=self.device)

        try:
            # Forward pass with or without gradient tracking
            context = torch.no_grad() if not with_grad else torch.enable_grad()
            with context:
                # Forward pass
                outputs = model(inputs)

                # Calculate per-sample code lengths
                code_lengths = encoding_fn(outputs, target, target_mask)
                # Sum the code lengths to get the total code length
                code_length = code_lengths.sum().item()

            # Move tensors back to CPU if needed to prevent GPU memory saturation
            inputs = self._move_to_cpu(inputs)
            target = self._move_to_cpu(target)
            target_mask = self._move_to_cpu(target_mask)

            return code_length, inputs, target, target_mask
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                raise RuntimeError(
                    "Device mismatch error. Please ensure your model handles device placement "
                    "for inputs and outputs in its forward method, and your loss function handles "
                    "device placement for its inputs."
                ) from e
            raise

    def update_step(self, model, batch, optim, encoding_fn=None):
        """
        Performs a single update step: calculates code length and updates the model.

        Args:
            model: PyTorch model.
            batch: Tuple containing inputs and target.
            optim: Optimizer.
            encoding_fn: Encoding function. If None, will use the encoder's encoding function.
                         This function should return per-sample code lengths.

        Returns:
            Tuple of (code_length, inputs, target, target_mask)
        """
        # Set encoding function if not provided
        if encoding_fn is None:
            encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        # Calculate loss with gradient tracking
        optim.zero_grad()
        inputs, target = batch[:2]

        # Move inputs and target to device
        inputs = self._move_to_device(inputs)
        target = self._move_to_device(target)

        # If there's a mask provided as 3rd element, use it
        if len(batch) > 2 and batch[2] is not None:
            target_mask = self._move_to_device(batch[2])
        else:
            # Default mask calculation
            target_mask = torch.ones_like(target, dtype=torch.bool, device=self.device)

        # Forward pass
        outputs = model(inputs)

        # Calculate per-sample code lengths
        code_lengths = encoding_fn(outputs, target, target_mask)
        # Sum the code lengths to get the total code length (loss)
        loss = code_lengths.sum()
        code_length = loss.item()

        # Backward pass and update
        loss.backward()
        optim.step()

        # Move tensors back to CPU if needed to prevent GPU memory saturation
        inputs = self._move_to_cpu(inputs)
        target = self._move_to_cpu(target)
        target_mask = self._move_to_cpu(target_mask)

        return code_length, inputs, target, target_mask

    def encode(self, dataset, set_name, epochs, learning_rate, batch_size, seed, stop_points, 
               patience=20, collate_fn=None, use_device_handling=False, shuffle=True, return_code_length_history=False,
               num_samples=None):
        """
        Prequential coding using block-based approach.

        Args:
            dataset: PyTorch dataset
            set_name: Name of the dataset (for logging)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            stop_points: List of points to split the dataset
            patience: Early stopping patience
            collate_fn: Collate function for DataLoader (if None, default collate will be used)
            use_device_handling: If True, will handle moving tensors to device (not recommended)
            shuffle: If True, will shuffle the training dataloader (default: True)
            return_code_length_history: If True, returns the full code_length history (default: False)
            num_samples: Number of samples to use from the dataset (if None, uses the full dataset)

        Returns:
            If return_code_length_history is False:
                Tuple of (model, code_length)
            If return_code_length_history is True:
                Tuple of (model, code_length, code_length_history)
        """
        # Set random seed before model initialization
        torch.manual_seed(seed)

        # Create a new model instance using _sample_model_class
        model = self._sample_model_class()

        # Initialize the optimizer
        optim = self._get_optimizer(model, learning_rate)

        # Set default encoding function if not provided
        encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        if use_device_handling:
            model.to(self.device)

        # Set up LR scheduler
        code_length = 0
        code_length_history = [] if return_code_length_history else None

        if stop_points[-1] != 1:
            stop_points.append(1)
        if stop_points[0] != 0:
            stop_points.insert(0, 0)

        chunk_sizes = []

        for stop_point_i, stop_point_j in zip(stop_points[1:], stop_points[:-1]):
            chunk_sizes.append(stop_point_i - stop_point_j)

        if num_samples is None:
            # Use the full dataset
            subset = dataset
        else:
            # Use a subset of the dataset
            subset = torch.utils.data.random_split(dataset, [min(num_samples, len(dataset)), max(0, len(dataset)-num_samples)])[0]
        chunks = torch.utils.data.random_split(subset, chunk_sizes)
        train_chunks = []
        for i in range(len(chunks)):
            train_chunks.append(torch.utils.data.ConcatDataset(chunks[:i+1]))

        init_params = deepcopy(model.state_dict())

        for train, eval in zip(train_chunks, chunks):
            train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
            val_dataloader = DataLoader(eval, batch_size=batch_size, collate_fn=collate_fn)
            for batch in val_dataloader:
                # Calculate code length for the batch
                loss_value, inputs, target, target_mask = self.calculate_code_length(model, batch, encoding_fn)
                code_length += loss_value
                if return_code_length_history:
                    code_length_history.append(loss_value)


            if train == train_chunks[-1]:
                break

            model.load_state_dict(deepcopy(init_params))
            model.train()
            no_improvement = 0
            best_loss = float('inf')
            for epoch in range(epochs):
                for batch in train_dataloader:
                    # Perform a single update step
                    loss_value, inputs, target, target_mask = self.update_step(model, batch, optim, encoding_fn)
                    if loss_value < best_loss:
                        no_improvement = 0
                        best_loss = loss_value
                    else:
                        no_improvement += 1
                        if no_improvement > patience:
                            break
                if no_improvement > patience:
                    break

                model.eval()

        print(f"Performance for {set_name}: Prequential code length: {code_length}")

        if return_code_length_history:
            return model, code_length, code_length_history
        else:
            return model, code_length


class MIRSEncoder(PrequentialEncoder):
    """
    Prequential encoder using MIRS (Multiple Independent Replay Streams) approach.
    """

    def __init__(self, model_class, loss_fn=None, device=None, optimizer_fn=None):
        """
        Initialize the MIRS encoder.

        Args:
            model_class: A PyTorch model class that will be instantiated for encoding
            loss_fn: Encoding function (if None, cross_entropy will be used)
                     This function should return per-sample code lengths
            device: Device to run the model on (if None, will use cuda if available, else cpu)
            optimizer_fn: Function to create optimizer (if None, Adam will be used)
        """
        super().__init__(model_class, loss_fn, device, optimizer_fn)

    def calculate_code_length(self, model, batch, beta=None, ema_params=None, use_beta=True, encoding_fn=None):
        """
        Calculates the code length for a single batch of data without updating the model.

        Args:
            model: PyTorch model.
            batch: Tuple containing inputs and target.
            beta: Trainable logit scaling factor (optional).
            ema_params: Dict for EMA parameters (optional). If provided, will use these parameters instead of model's current parameters.
            use_beta: Whether to use beta scaling.
            encoding_fn: Encoding function. If None, will use the encoder's encoding function.
                         This function should return per-sample code lengths.
            with_grad: Whether to calculate gradients. If False, will use torch.no_grad().

        Returns:
            Tuple of (code_length, inputs, target, target_mask)
        """
        # Set encoding function if not provided
        if encoding_fn is None:
            encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        # Save current parameters if ema_params is provided
        original_params = {}
        if ema_params is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    original_params[name] = param.clone().detach()
                    # Temporarily replace with EMA parameters
                    param.data.copy_(ema_params[name].data)

        # Handle different dataset formats
        inputs, target = batch[:2]

        # Move inputs and target to device
        inputs = self._move_to_device(inputs)
        target = self._move_to_device(target)

        # If there's a mask provided as 3rd element, use it
        if len(batch) > 2 and batch[2] is not None:
            target_mask = self._move_to_device(batch[2])
        else:
            # Default mask calculation
            target_mask = (target != 0).to(self.device)

        try:
            # Forward pass with beta scaling if enabled
            if use_beta and beta is not None:
                outputs = model(inputs) * F.softplus(beta)
            else:
                outputs = model(inputs)

            # Calculate per-sample code lengths
            code_lengths = encoding_fn(outputs, target, target_mask)

            # Restore original parameters if ema_params was provided
            if ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.data.copy_(original_params[name].data)

            # Move tensors back to CPU if needed to prevent GPU memory saturation
            inputs = self._move_to_cpu(inputs)
            target = self._move_to_cpu(target)
            target_mask = self._move_to_cpu(target_mask)

            return code_lengths, inputs, target, target_mask
        except RuntimeError as e:
            # Restore original parameters if an exception occurred
            if ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.data.copy_(original_params[name].data)

            if "Expected all tensors to be on the same device" in str(e):
                raise RuntimeError(
                    "Device mismatch error. Please ensure your model handles device placement "
                    "for inputs and outputs in its forward method, and your loss function handles "
                    "device placement for its inputs."
                ) from e
            raise
        except Exception as e:
            # Restore original parameters if an exception occurred
            if ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.data.copy_(original_params[name].data)
            raise

    def training_update_step(self, model, batch, replay_streams, beta, alpha, ema_params, trained_params,
                             optim, beta_optim, encoding_fn):
        """
        Performs a single update step: calculates code length, updates the model, and updates EMA parameters.

        Args:
            model: PyTorch model.
            batch: Tuple containing inputs and target.
            replay_streams: a ReplayStreams object used to retrain on past samples
            beta: Trainable logit scaling factor.
            alpha: EMA smoothing factor.
            ema_params, trained_params: Dicts for EMA and temporary weight storage.
            optim, beta_optim: Optimizers.
            encoding_fn: Encoding function. This function should return per-sample code lengths.
            use_beta: Whether to use beta scaling.

        Returns:
            Tuple of (code_length, inputs, target, target_mask)
        """
        # Save current parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                trained_params[name] = param.clone().detach()

        if beta is not None:
            use_beta = True
        else:
            use_beta = False

        optim.zero_grad()
        if use_beta:
            beta_optim.zero_grad()

        new_code_length = None

        # Calculate code length for new batch using ema/temperature if applicable
        if ema_params is not None or use_beta:
            code_lengths, inputs, target, target_mask = self.calculate_code_length(
                model, batch, beta, ema_params, use_beta, encoding_fn
            )
            training_loss = code_lengths.sum()  # Sum the code lengths to get the total loss
            new_code_length = training_loss.detach()
            training_loss.backward()
            if use_beta:
                beta_optim.step()

        optim.zero_grad()  # Zero gradients before backward pass
        if use_beta:
            beta_optim.zero_grad()  # Zero beta gradients before backward pass
        code_lengths, inputs, target, target_mask = self.calculate_code_length(
            model, batch, None, None, None, encoding_fn
        )
        training_loss = code_lengths.sum()  # Sum the code lengths to get the total loss
        if new_code_length is None:
            new_code_length = training_loss.detach()

        training_loss.backward()
        optim.step()

        # Update EMA parameters with no_grad to avoid in-place operations affecting the backward graph
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Create a new tensor for the updated EMA value instead of modifying in-place
                ema_params[name] = ema_params[name].clone() * (1-alpha) + param.clone() * alpha

        replay_streams.update(batch)

        for index, sample in replay_streams.sample():
            optim.zero_grad()  # Zero gradients before backward pass
            code_lengths, inputs, target, target_mask = self.calculate_code_length(
                model, sample, None, None, None, encoding_fn
            )

            training_loss = code_lengths.sum()  # Sum the code lengths to get the total loss

            training_loss.backward()
            optim.step()

            # Update EMA parameters with no_grad to avoid in-place operations affecting the backward graph
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # Create a new tensor for the updated EMA value instead of modifying in-place
                    ema_params[name] = ema_params[name].clone() * (1 - alpha) + param.clone() * alpha

        # Move tensors back to CPU if needed to prevent GPU memory saturation
        inputs = self._move_to_cpu(inputs)
        target = self._move_to_cpu(target)
        target_mask = self._move_to_cpu(target_mask)

        return new_code_length, inputs, target, target_mask

    def encode(self, dataset, set_name, n_replay_streams, learning_rate, batch_size, seed, alpha=0.1,
               collate_fn=None, use_device_handling=False, use_beta=True, use_ema=True, shuffle=True, 
               return_code_length_history=False, num_samples=None, replay_streams=None):
        """
        Prequential coding using MIRS (Mini-batch Incremental Replay Streams) approach.
        Initializes training and encodes a whole dataset.

        Args:
            dataset: PyTorch dataset
            set_name: Name of the dataset (for logging)
            n_replay_streams: Number of replay streams
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            alpha: EMA parameter for model parameters (only used if use_ema is True)
            collate_fn: Collate function for DataLoader (if None, default collate will be used)
            use_device_handling: If True, will handle moving tensors to device (not recommended)
            use_beta: Whether to use beta scaling
            use_ema: Whether to use EMA smoothing
            shuffle: If True, will shuffle the dataloader (default: True)
            return_code_length_history: If True, returns the full code_length history (default: False)
            num_samples: Number of samples to use from the dataset (if None, uses the full dataset)
            replay_streams: Optional ReplayStreams instance. If None, a new one will be created.

        Returns:
            If return_code_length_history is False:
                Tuple of (model, code_length, ema_params, beta, replay_streams)
            If return_code_length_history is True:
                Tuple of (model, code_length, code_length_history, ema_params, beta, replay_streams)
        """
        # Set random seed before model initialization
        torch.manual_seed(seed)
        random.seed(seed)

        # Create a new model instance using _sample_model_class
        model = self._sample_model_class()

        # Initialize the optimizer
        optim = self._get_optimizer(model, learning_rate)

        # Set default encoding function if not provided
        encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        beta = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        beta_optim = torch.optim.Adam([beta], lr=learning_rate)
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        if use_device_handling:
            model.to(self.device)

        # Set up LR scheduler
        code_length = 0
        code_length_history = [] if return_code_length_history else None

        gen = torch.Generator()
        gen.manual_seed(seed)

        if num_samples is None:
            # Use the full dataset
            if shuffle:
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=gen)
            else:
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        else:
            # Use a subset of the dataset
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples, generator=gen)
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
            else:
                # Use SequentialSampler if shuffle is False
                sampler = torch.utils.data.SequentialSampler(dataset)
                # Limit to num_samples to maintain consistency with the shuffled version
                subset = torch.utils.data.Subset(dataset, list(range(min(num_samples, len(dataset)))))
                dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

        # Initialize or use provided ReplayStreams
        if replay_streams is None:
            # Create a new ReplayStreams instance with the default reset probability function
            replay_streams = ReplayStreams(n_replay_streams)

        model.train()
        ema_params = {}
        trained_params = {}
        for name, param in model.named_parameters():
            ema_params[name] = param.clone().detach()
            trained_params[name] = param.clone().detach()

        for batch in dataloader:

            # Perform a single update step
            code_len, inputs, target, target_mask = self.training_update_step(
                model, batch, replay_streams, beta, alpha, ema_params, trained_params,
                optim, beta_optim, encoding_fn
            )
            code_length += code_len
            if return_code_length_history:
                code_length_history.append(code_len)

        print(f"Performance for {set_name}: Prequential code length: {code_length}")

        # Move beta and ema_params back to CPU if needed to prevent GPU memory saturation
        if self.device != 'cpu' and self.device != torch.device('cpu'):
            beta.data = beta.data.cpu()
            # Move EMA parameters to CPU
            for name in ema_params:
                ema_params[name] = ema_params[name].cpu()

        if return_code_length_history:
            return model, code_length, code_length_history, ema_params, beta, replay_streams
        else:
            return model, code_length, ema_params, beta, replay_streams

    def encode_with_model(self, model, ema_params, beta, dataset, set_name, n_replay_streams, 
                         learning_rate, batch_size, seed, alpha=0.1, collate_fn=None, 
                         use_device_handling=False, use_beta=True, use_ema=True, shuffle=True, 
                         return_code_length_history=False, num_samples=None, replay_streams=None):
        """
        Prequential coding using MIRS approach with an existing model and parameters.
        This allows for continuing training from a previous state.

        Args:
            model: Existing PyTorch model
            ema_params: Dictionary of EMA parameters
            beta: Beta parameter for scaling
            dataset: PyTorch dataset
            set_name: Name of the dataset (for logging)
            n_replay_streams: Number of replay streams
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            alpha: EMA parameter for model parameters (only used if use_ema is True)
            collate_fn: Collate function for DataLoader (if None, default collate will be used)
            use_device_handling: If True, will handle moving tensors to device (not recommended)
            use_beta: Whether to use beta scaling
            use_ema: Whether to use EMA smoothing
            shuffle: If True, will shuffle the dataloader (default: True)
            return_code_length_history: If True, returns the full code_length history (default: False)
            num_samples: Number of samples to use from the dataset (if None, uses the full dataset)
            replay_streams: Optional ReplayStreams instance. If None, a new one will be created.

        Returns:
            If return_code_length_history is False:
                Tuple of (model, code_length, ema_params, beta, replay_streams)
            If return_code_length_history is True:
                Tuple of (model, code_length, code_length_history, ema_params, beta, replay_streams)
        """
        # Set random seed
        torch.manual_seed(seed)
        random.seed(seed)

        # Initialize the optimizer
        optim = self._get_optimizer(model, learning_rate)

        # Set default encoding function if not provided
        encoding_fn = self.encoding_fn if self.encoding_fn is not None else self._get_default_encoding_fn()

        # Initialize beta optimizer
        beta_optim = torch.optim.Adam([beta], lr=learning_rate)
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        if use_device_handling:
            model.to(self.device)
            beta.data = beta.data.to(self.device)

        # Set up LR scheduler
        code_length = 0
        code_length_history = [] if return_code_length_history else None

        gen = torch.Generator()
        gen.manual_seed(seed)

        if num_samples is None:
            # Use the full dataset
            if shuffle:
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=gen)
            else:
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        else:
            # Use a subset of the dataset
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples, generator=gen)
                dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
            else:
                # Use SequentialSampler if shuffle is False
                sampler = torch.utils.data.SequentialSampler(dataset)
                # Limit to num_samples to maintain consistency with the shuffled version
                subset = torch.utils.data.Subset(dataset, list(range(min(num_samples, len(dataset)))))
                dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)

        # Initialize or use provided ReplayStreams
        if replay_streams is None:
            # Create a new ReplayStreams instance with the default reset probability function
            replay_streams = ReplayStreams(n_replay_streams)

        model.train()
        trained_params = {}
        for name, param in model.named_parameters():
            trained_params[name] = param.clone().detach()

        for batch in dataloader:

            # Perform a single update step
            code_len, inputs, target, target_mask = self.training_update_step(
                model, batch, replay_streams, beta, alpha, ema_params, trained_params,
                optim, beta_optim, encoding_fn
            )
            code_length += code_len
            if return_code_length_history:
                code_length_history.append(code_len)

        print(f"Performance for {set_name}: Prequential code length: {code_length}")

        # Move beta and ema_params back to CPU if needed to prevent GPU memory saturation
        if self.device != 'cpu' and self.device != torch.device('cpu'):
            beta.data = beta.data.cpu()
            # Move EMA parameters to CPU
            for name in ema_params:
                ema_params[name] = ema_params[name].cpu()

        if return_code_length_history:
            return model, code_length, code_length_history, ema_params, beta, replay_streams
        else:
            return model, code_length, ema_params, beta, replay_streams