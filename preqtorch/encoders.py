import inspect
import random

import torch
from torch.func import functional_call

from .batches import move_to_device
from .replay import ReplayBuffer, ReplayStreams, ReplayingDataLoader
from .results import EncoderResult
from .utils import ModelClass


class EncoderState:
    def __init__(self, model, optim, beta, beta_optim, ema_params, trained_params, loss_fn):
        self.model = model
        self.optim = optim
        self.beta = beta
        self.beta_optim = beta_optim
        self.ema_params = ema_params
        self.trained_params = trained_params
        self.loss_fn = loss_fn
        self.code_length = 0
        self.history = []


class PrequentialEncoder:
    def __init__(self, model_class: ModelClass, device=None, optimizer_fn=None, pin_memory=False):
        self.model_class = model_class
        if device is not None:
            self.device = device
        elif hasattr(model_class, "device"):
            self.device = model_class.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hasattr(self.model_class, 'to'):
            self.model_class.to(self.device)
        self.optimizer_fn = optimizer_fn
        self.pin_memory = pin_memory

    def to(self, device):
        self.device = device
        if hasattr(self.model_class, 'to'):
            self.model_class.to(device)
        return self

    def _get_optimizer(self, model, learning_rate):
        if self.optimizer_fn is None:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        return self.optimizer_fn(model.parameters(), lr=learning_rate)

    def _sample_model_class(self):
        return self.model_class.initialize()

    @staticmethod
    def _callable_accepts_n_positional_args(fn, n_required):
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        required_positional = [p for p in positional if p.default is inspect._empty]
        return has_varargs or (len(positional) >= n_required and len(required_positional) <= n_required)

    def _validate_model_forward_contract(self, model):
        if not self._callable_accepts_n_positional_args(model.forward, 1):
            signature = inspect.signature(model.forward)
            raise TypeError(
                "Model forward contract mismatch. Expected forward(batch). "
                f"Got forward{signature}."
            )

    def _validate_loss_fn_contract(self, loss_fn):
        if loss_fn is None:
            raise ValueError("A loss_fn must be provided. Expected loss_fn(batch, output).")
        if not self._callable_accepts_n_positional_args(loss_fn, 2):
            signature = inspect.signature(loss_fn)
            name = getattr(loss_fn, '__name__', 'loss_fn')
            raise TypeError(
                "Loss function contract mismatch. Expected fn(batch, output). "
                f"Got {name}{signature}."
            )

    def _prepare_batch(self, batch, use_device_handling=True):
        if use_device_handling:
            return move_to_device(batch, self.device, non_blocking=self.pin_memory)
        return batch


class BlockEncoder(PrequentialEncoder):
    def encode(self, train_dataloader, eval_dataloaders, set_name, seed, loss_fn=None,
               learning_rate=1e-4, epochs=50, patience=20, use_device_handling=True):
        self._validate_loss_fn_contract(loss_fn)
        torch.manual_seed(seed)
        random.seed(seed)

        model = self._sample_model_class()
        model.to(self.device)
        optim = self._get_optimizer(model, learning_rate)

        state = EncoderState(model=model, optim=optim, beta=None, beta_optim=None,
                             ema_params=None, trained_params=None, loss_fn=loss_fn)

        if len(train_dataloader) != len(eval_dataloaders):
            raise ValueError("train_dataloader and eval_dataloaders must have same length")

        initial_weights = {name: value.detach().clone() for name, value in state.model.state_dict().items()}

        for i, (train_loader, eval_loader) in enumerate(zip(train_dataloader, eval_dataloaders)):
            self.eval_code_length(state, eval_loader, loss_fn, use_device_handling=use_device_handling)
            if i == len(train_dataloader) - 1:
                break
            state.model.load_state_dict(initial_weights)
            self.train_until_patience(state, train_loader, patience, epochs, use_device_handling=use_device_handling)

        print(f"Performance for {set_name}: Prequential code length: {state.code_length}")
        return EncoderResult(state.model, state.code_length, state.history)

    def calculate_code_length(self, model, batch, loss_fn, use_device_handling=True):
        self._validate_model_forward_contract(model)
        self._validate_loss_fn_contract(loss_fn)
        batch = self._prepare_batch(batch, use_device_handling=use_device_handling)
        output = model(batch)
        code_lengths = loss_fn(batch, output)
        return code_lengths, batch, output

    def eval_code_length(self, state, dataloader, loss_fn=None, use_device_handling=True):
        loss_fn = loss_fn or state.loss_fn
        model = state.model
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                code_lengths, _, _ = self.calculate_code_length(model, batch, loss_fn, use_device_handling=use_device_handling)
                value = code_lengths.sum().item()
                state.code_length += value
                state.history.append(value)

    def train_until_patience(self, state, train_dataloader, patience, epochs, use_device_handling=True):
        best_loss = float('inf')
        no_improvement = 0
        model = state.model
        optim = state.optim
        model.train()

        for _ in range(epochs):
            for batch in train_dataloader:
                optim.zero_grad()
                code_lengths, _, _ = self.calculate_code_length(model, batch, state.loss_fn, use_device_handling=use_device_handling)
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
    def encode(self, dataloader, set_name, n_replay_samples, loss_fn=None, learning_rate=1e-4,
               seed=42, alpha=0.1, collate_fn=None, pin_memory=None, use_device_handling=True, use_beta=True,
               use_ema=True, shuffle=True, replay_type="buffer"):
        self._validate_loss_fn_contract(loss_fn)
        if pin_memory is None:
            pin_memory = self.pin_memory
        self.pin_memory = pin_memory

        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        if batch_size is None:
            raise ValueError("Dataloader must define batch_size")

        if collate_fn is None:
            collate_fn = dataloader.collate_fn

        state, replay_loader = self.initialize(
            dataset, batch_size, seed, n_replay_samples, replay_type,
            None, learning_rate, alpha, collate_fn, shuffle, use_beta, use_ema, loss_fn,
            pin_memory=pin_memory)

        for batch in replay_loader:
            self.step(batch, replay_loader, state, alpha, use_device_handling=use_device_handling)

        model, code_length, history, ema_params, beta = self.finalize(state)
        return EncoderResult(model, code_length, history, ema_params=ema_params, beta=beta, replay=replay_loader.replay)

    def initialize(self, dataset, batch_size, seed, n_replay_samples, replay_type="buffer",
                   model=None, learning_rate=1e-4, alpha=0.1, collate_fn=None, shuffle=True, use_beta=True,
                   use_ema=False, loss_fn=None, pin_memory=False):
        self._validate_loss_fn_contract(loss_fn)
        torch.manual_seed(seed)
        random.seed(seed)

        model = model or self._sample_model_class()
        model.to(self.device)

        optim = self._get_optimizer(model, learning_rate)
        if use_beta:
            beta = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            beta_optim = torch.optim.Adam([beta], lr=learning_rate)
        else:
            beta = None
            beta_optim = None

        if replay_type == "streams":
            replay_impl = ReplayStreams(dataset, batch_size=batch_size, n_streams=n_replay_samples, collate_fn=collate_fn)
        elif replay_type == "buffer":
            replay_impl = ReplayBuffer(dataset, batch_size=batch_size, n_samples=n_replay_samples, collate_fn=collate_fn)
        else:
            raise ValueError("replay_type must be 'streams' or 'buffer'")

        replay_loader = ReplayingDataLoader(dataset, batch_size=batch_size, replay=replay_impl,
                                            collate_fn=collate_fn, shuffle=shuffle, pin_memory=pin_memory)

        if use_ema:
            ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}
            trained_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        else:
            ema_params = None
            trained_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        state = EncoderState(model, optim, beta, beta_optim, ema_params, trained_params, loss_fn=loss_fn)
        return state, replay_loader

    @staticmethod
    def _scale_output(output, beta):
        if not isinstance(output, torch.Tensor):
            raise TypeError("use_beta=True requires the model output to be a torch.Tensor")
        return output * torch.nn.functional.softplus(beta)

    def step(self, batch, replay_loader, state, alpha=0.1, use_device_handling=True):
        model = state.model
        beta = state.beta

        use_beta = beta is not None
        state.optim.zero_grad()
        if use_beta:
            state.beta_optim.zero_grad()

        code_lengths, _, _ = self.calculate_code_length(state, batch, use_device_handling=use_device_handling)
        loss = code_lengths.sum()
        state.code_length += loss.detach()
        state.history.append(loss.detach())
        if use_beta:
            loss.backward()
            state.beta_optim.step()
            state.beta_optim.zero_grad()
        if use_beta or state.ema_params is not None:
            state.optim.zero_grad()
            code_lengths, _, _ = self.calculate_code_length(state, batch, False, False, use_device_handling=use_device_handling)
            loss = code_lengths.sum()
            loss.backward()
        state.optim.step()
        state.optim.zero_grad()
        if use_beta:
            state.beta_optim.zero_grad()

        if state.ema_params is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    state.ema_params[name] = state.ema_params[name] * (1 - alpha) + param * alpha

        for _, replay_batch in replay_loader.sample_replay():
            state.optim.zero_grad()
            code_lengths, _, _ = self.calculate_code_length(state, replay_batch, False, False, use_device_handling=use_device_handling)
            loss = code_lengths.sum()
            loss.backward()
            state.optim.step()
            if state.ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        state.ema_params[name] = state.ema_params[name] * (1 - alpha) + param * alpha

    def calculate_code_length(self, state, batch, use_ema=True, use_beta=True, use_device_handling=True):
        loss_fn = state.loss_fn
        model = state.model
        beta = state.beta
        ema_params = state.ema_params
        self._validate_model_forward_contract(model)
        self._validate_loss_fn_contract(loss_fn)

        batch = self._prepare_batch(batch, use_device_handling=use_device_handling)

        if ema_params is not None and use_ema:
            params_and_buffers = {name: buffer for name, buffer in model.named_buffers()}
            params_and_buffers.update({name: param for name, param in model.named_parameters()})
            params_and_buffers.update({name: value for name, value in ema_params.items() if name in params_and_buffers})
            output = functional_call(model, params_and_buffers, (batch,))
        else:
            output = model(batch)

        if beta is not None and use_beta:
            output = self._scale_output(output, beta)

        code_lengths = loss_fn(batch, output)
        return code_lengths, batch, output

    def finalize(self, state):
        if self.device != 'cpu':
            if state.beta is not None:
                state.beta.data = state.beta.data.cpu()
            if state.ema_params is not None:
                for name in state.ema_params:
                    state.ema_params[name] = state.ema_params[name].cpu()

        return state.model, state.code_length, state.history, state.ema_params, state.beta
