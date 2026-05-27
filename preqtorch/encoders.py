import math
import random

import torch
import torch.nn.functional as F
from torch.func import functional_call

from .batches import PrequentialBatch
from .replay import ReplayBuffer, ReplayStreams, ReplayingDataLoader
from .results import EncoderResult
from .utils import ModelClass

LOG2 = math.log(2)


class EncoderState:
    def __init__(self, model, optim, beta, beta_optim, ema_params, trained_params, encoding_fn=None):
        self.model = model
        self.optim = optim
        self.beta = beta
        self.beta_optim = beta_optim
        self.ema_params = ema_params
        self.trained_params = trained_params
        self.encoding_fn = encoding_fn
        self.code_length = 0
        self.history = []


class PrequentialEncoder:
    def __init__(self, model_class: ModelClass, device=None, optimizer_fn=None, pin_memory=False):
        self.model_class = model_class
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self.model_class, 'to'):
            self.model_class.to(self.device)
        self.optimizer_fn = optimizer_fn
        self.pin_memory = pin_memory
        self._default_mask_cache = {}

    def to(self, device):
        self.device = device
        if hasattr(self.model_class, 'to'):
            self.model_class.to(device)
        return self

    def _get_default_encoding_fn(self):
        def encoding_fn(outputs, targets, output_mask, target_mask):
            log2 = torch.tensor(LOG2, device=outputs.device)
            return torch.nn.functional.cross_entropy(outputs[output_mask], targets[target_mask], reduction='none') / log2

        return encoding_fn

    def _get_optimizer(self, model, learning_rate):
        if self.optimizer_fn is None:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        return self.optimizer_fn(model.parameters(), lr=learning_rate)

    def _sample_model_class(self):
        return self.model_class.initialize()

    def _get_default_mask(self, tensor):
        key = (tuple(tensor.shape), tensor.device, torch.bool)
        mask = self._default_mask_cache.get(key)
        if mask is None or (not torch.is_inference_mode_enabled() and mask.is_inference()):
            mask = torch.ones_like(tensor, dtype=torch.bool, device=tensor.device)
            self._default_mask_cache[key] = mask
        return mask

    def _normalize_batch(self, batch, use_device_handling=True):
        if isinstance(batch, PrequentialBatch):
            if use_device_handling:
                return batch.to(self.device, non_blocking=self.pin_memory)
            return batch
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                inputs, targets = batch
                output_mask = self._get_default_mask(targets)
                target_mask = self._get_default_mask(targets)
            elif len(batch) == 3:
                inputs, targets, mask = batch
                output_mask = mask
                target_mask = mask
            elif len(batch) == 4:
                inputs, targets, output_mask, target_mask = batch
            else:
                raise ValueError("Unsupported batch format")
            normalized = PrequentialBatch(inputs, targets, output_mask, target_mask)
            if use_device_handling:
                return normalized.to(self.device, non_blocking=self.pin_memory)
            return normalized
        raise TypeError("Unsupported batch type")


class BlockEncoder(PrequentialEncoder):
    def encode(self, train_dataloader, eval_dataloaders, set_name, seed, learning_rate=1e-4, epochs=50, patience=20, collate_fn=None, use_device_handling=True, encoding_fn=None):
        torch.manual_seed(seed)
        random.seed(seed)

        model = self._sample_model_class()
        model.to(self.device)
        optim = self._get_optimizer(model, learning_rate)
        encoding_fn = encoding_fn or self._get_default_encoding_fn()

        state = EncoderState(model=model, optim=optim, beta=None, beta_optim=None, ema_params=None, trained_params=None, encoding_fn=encoding_fn)

        if len(train_dataloader) != len(eval_dataloaders):
            raise ValueError("train_dataloader and eval_dataloaders must have same length")

        if collate_fn is not None:
            def _with_collate(loader):
                return loader if loader.collate_fn is collate_fn else torch.utils.data.DataLoader(
                    dataset=loader.dataset,
                    batch_size=loader.batch_size,
                    shuffle=False,
                    sampler=loader.sampler,
                    num_workers=loader.num_workers,
                    pin_memory=loader.pin_memory,
                    drop_last=loader.drop_last,
                    timeout=loader.timeout,
                    worker_init_fn=loader.worker_init_fn,
                    collate_fn=collate_fn,
                )

            train_dataloader = [_with_collate(loader) for loader in train_dataloader]
            eval_dataloaders = [_with_collate(loader) for loader in eval_dataloaders]

        initial_weights = {name: value.detach().clone() for name, value in state.model.state_dict().items()}

        for i, (train_loader, eval_loader) in enumerate(zip(train_dataloader, eval_dataloaders)):
            self.eval_code_length(state, eval_loader, encoding_fn, use_device_handling=use_device_handling)
            if i == len(train_dataloader) - 1:
                break
            state.model.load_state_dict(initial_weights)
            self.train_until_patience(state, train_loader, patience, epochs, use_device_handling=use_device_handling)

        print(f"Performance for {set_name}: Prequential code length: {state.code_length}")
        return EncoderResult(state.model, state.code_length, state.history)

    def calculate_code_length(self, model, batch, encoding_fn=None, use_device_handling=True):
        encoding_fn = encoding_fn or self._get_default_encoding_fn()
        spec = self._normalize_batch(batch, use_device_handling=use_device_handling)
        outputs = model(spec.inputs)
        code_lengths = encoding_fn(outputs, spec.targets, spec.output_mask, spec.target_mask)
        return code_lengths, spec.inputs, spec.targets, spec.target_mask, spec.output_mask

    def eval_code_length(self, state, dataloader, encoding_fn=None, use_device_handling=True):
        encoding_fn = encoding_fn or state.encoding_fn or self._get_default_encoding_fn()
        model = state.model
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                code_lengths, _, _, _, _ = self.calculate_code_length(model, batch, encoding_fn, use_device_handling=use_device_handling)
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
                encoding_fn = state.encoding_fn or self._get_default_encoding_fn()
                optim.zero_grad()
                code_lengths, _, _, _, _ = self.calculate_code_length(model, batch, encoding_fn, use_device_handling=use_device_handling)
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
    def encode(self, dataloader, set_name, n_replay_samples, learning_rate=1e-4,
               seed=42, alpha=0.1, collate_fn=None, pin_memory=None, use_device_handling=True, use_beta=True,
               use_ema=True, shuffle=True, replay_type="buffer", encoding_fn=None):
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
            None, learning_rate, alpha, collate_fn, shuffle, use_beta, use_ema, encoding_fn,
            pin_memory=pin_memory)

        for batch in replay_loader:
            self.step(batch, replay_loader, state, alpha, use_device_handling=use_device_handling)

        model, code_length, history, ema_params, beta = self.finalize(state)
        return EncoderResult(model, code_length, history, ema_params=ema_params, beta=beta, replay=replay_loader.replay)

    def initialize(self, dataset, batch_size, seed, n_replay_samples, replay_type="buffer",
                   model=None, learning_rate=1e-4, alpha=0.1, collate_fn=None, shuffle=True, use_beta=True,
                   use_ema=False, encoding_fn=None, pin_memory=False):
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

        if encoding_fn is None:
            encoding_fn = self._get_default_encoding_fn()

        state = EncoderState(model, optim, beta, beta_optim, ema_params, trained_params, encoding_fn=encoding_fn)
        return state, replay_loader

    def step(self, batch, replay_loader, state, alpha=0.1, use_device_handling=True):
        model = state.model
        beta = state.beta

        use_beta = beta is not None
        state.optim.zero_grad()
        if use_beta:
            state.beta_optim.zero_grad()

        code_lengths, _, _, _, _ = self.calculate_code_length(state, batch, use_device_handling=use_device_handling)
        loss = code_lengths.sum()
        state.code_length += loss.detach()
        state.history.append(loss.detach())
        if use_beta:
            loss.backward()
            state.beta_optim.step()
            state.beta_optim.zero_grad()
        if use_beta or state.ema_params is not None:
            state.optim.zero_grad()
            code_lengths, _, _, _, _ = self.calculate_code_length(state, batch, False, False, use_device_handling=use_device_handling)
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
            code_lengths, _, _, _, _ = self.calculate_code_length(state, replay_batch, False, False, use_device_handling=use_device_handling)
            loss = code_lengths.sum()
            loss.backward()
            state.optim.step()
            if state.ema_params is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        state.ema_params[name] = state.ema_params[name] * (1 - alpha) + param * alpha

    def calculate_code_length(self, state, batch, use_ema=True, use_beta=True, use_device_handling=True):
        encoding_fn = state.encoding_fn or self._get_default_encoding_fn()
        model = state.model
        beta = state.beta
        ema_params = state.ema_params

        spec = self._normalize_batch(batch, use_device_handling=use_device_handling)
        inputs = spec.inputs
        target = spec.targets
        output_mask = spec.output_mask
        target_mask = spec.target_mask

        if ema_params is not None and use_ema:
            params_and_buffers = {name: buffer for name, buffer in model.named_buffers()}
            params_and_buffers.update({name: param for name, param in model.named_parameters()})
            params_and_buffers.update({name: value for name, value in ema_params.items() if name in params_and_buffers})
            outputs = functional_call(model, params_and_buffers, (inputs,))
        else:
            outputs = model(inputs)

        if beta is not None and use_beta:
            outputs = outputs * F.softplus(beta)

        code_lengths = encoding_fn(outputs, target, output_mask, target_mask)
        return code_lengths, inputs, target, target_mask, output_mask

    def finalize(self, state):
        if self.device != 'cpu':
            if state.beta is not None:
                state.beta.data = state.beta.data.cpu()
            if state.ema_params is not None:
                for name in state.ema_params:
                    state.ema_params[name] = state.ema_params[name].cpu()

        return state.model, state.code_length, state.history, state.ema_params, state.beta
