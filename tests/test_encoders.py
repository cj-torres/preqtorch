import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, SubsetRandomSampler, TensorDataset

from preqtorch import BlockEncoder, EncoderResult, MIREncoder, ModelClass, Replay
from preqtorch.encoders import EncoderState


class BatchLinear(torch.nn.Module):
    def __init__(self, input_size=2, output_size=2):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        inputs, _ = batch
        return self.linear(inputs.float())


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        inputs, _ = batch
        return inputs * self.weight


def zero_scalar_model(model):
    with torch.no_grad():
        model.weight.zero_()
    return model


def negative_output_loss(batch, output):
    return -output


class RecordingReplay(Replay):
    def __init__(self, dataset, collate_fn=None):
        super().__init__(dataset, collate_fn)
        self.updated_indices = []
        self.sample_calls = 0

    def update(self, new_indices):
        self.updated_indices.append(list(new_indices))

    def sample(self):
        self.sample_calls += 1
        return []


class VariableBatchSampler:
    def __iter__(self):
        yield [2, 0]
        yield [1]

    def __len__(self):
        return 2


class SimpleIterableDataset(IterableDataset):
    def __iter__(self):
        yield torch.ones(2), torch.tensor(0)


def code_length_loss(batch, output):
    _, targets = batch
    return torch.nn.functional.cross_entropy(output, targets, reduction="none")


def make_loader(batch_size=2):
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def test_encoder_result_fields():
    result = EncoderResult(model="m", code_length=1.0, history=[1, 2, 3])
    assert result.model == "m"
    assert result.code_length == 1.0
    assert result.history == [1, 2, 3]

    replay_result = EncoderResult(model="m", code_length=1.0, history=[], ema_params={}, beta="b", replay="r")
    assert replay_result.replay == "r"


def test_block_encoder_uses_model_batch_and_loss_fn_contract():
    model_class = ModelClass(BatchLinear, device="cpu")
    encoder = BlockEncoder(model_class=model_class, device="cpu")
    loader = make_loader()

    result = encoder.encode(
        train_dataloader=[loader],
        eval_dataloaders=[loader],
        set_name="opaque batches",
        seed=42,
        loss_fn=code_length_loss,
        epochs=1,
        patience=1,
    )

    assert isinstance(result, EncoderResult)
    assert result.code_length > 0
    assert len(result.history) == len(loader)


def test_block_calculate_code_length_passes_whole_batch_to_model_and_loss():
    model = BatchLinear()
    encoder = BlockEncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")
    batch = next(iter(make_loader()))
    seen = {}

    def loss_fn(batch_arg, output):
        seen["batch"] = batch_arg
        seen["output"] = output
        return code_length_loss(batch_arg, output)

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        model,
        batch,
        loss_fn=loss_fn,
        use_device_handling=False,
    )

    assert model.last_batch is batch
    assert seen["batch"] is batch
    assert seen["output"] is output
    assert returned_batch is batch
    assert code_lengths.shape == batch[1].shape


def test_block_encoder_uses_fresh_optimizer_for_each_weight_restart():
    loader = DataLoader(
        TensorDataset(torch.ones(1), torch.zeros(1)),
        batch_size=1,
        shuffle=False,
    )
    optimizers = []

    def optimizer_fn(parameters, lr):
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
        optimizers.append(optimizer)
        return optimizer

    encoder = BlockEncoder(
        ModelClass(ScalarModel, device="cpu", init_func=zero_scalar_model),
        device="cpu",
        optimizer_fn=optimizer_fn,
    )
    result = encoder.encode(
        train_dataloader=[loader, loader, loader],
        eval_dataloaders=[loader, loader, loader],
        set_name="optimizer restarts",
        seed=42,
        loss_fn=negative_output_loss,
        learning_rate=1.0,
        epochs=1,
        patience=1,
    )

    assert len(optimizers) == 2
    assert result.model.weight.item() == 1.0


def test_mir_encoder_uses_model_batch_and_loss_fn_contract():
    model_class = ModelClass(BatchLinear, device="cpu")
    encoder = MIREncoder(model_class=model_class, device="cpu")
    loader = make_loader(batch_size=2)

    result = encoder.encode(
        dataloader=loader,
        set_name="opaque batches mir",
        n_replay_samples=1,
        loss_fn=code_length_loss,
        learning_rate=1e-3,
        seed=42,
        use_beta=False,
        use_ema=False,
    )

    assert isinstance(result, EncoderResult)
    assert isinstance(result.code_length, float)
    assert result.code_length > 0
    assert all(isinstance(value, float) for value in result.history)
    assert len(result.history) == len(loader)
    assert result.replay is not None


def test_mir_encoder_updates_model_without_beta_or_ema():
    loader = DataLoader(
        TensorDataset(torch.ones(1), torch.zeros(1)),
        batch_size=1,
        shuffle=False,
    )
    encoder = MIREncoder(
        ModelClass(ScalarModel, device="cpu", init_func=zero_scalar_model),
        device="cpu",
        optimizer_fn=lambda parameters, lr: torch.optim.SGD(parameters, lr=lr),
    )

    result = encoder.encode(
        dataloader=loader,
        set_name="plain optimizer update",
        n_replay_samples=0,
        loss_fn=negative_output_loss,
        learning_rate=1.0,
        use_beta=False,
        use_ema=False,
    )

    assert result.model.weight.item() == 1.0


def test_mir_encoder_preserves_source_batch_sampler():
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(inputs, targets)
    sampler = SubsetRandomSampler(
        [0, 2],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(sampler, batch_size=2, drop_last=False),
    )
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")

    result = encoder.encode(
        dataloader=loader,
        set_name="source batch sampler",
        n_replay_samples=1,
        loss_fn=code_length_loss,
        learning_rate=1e-3,
        seed=42,
        use_beta=False,
        use_ema=False,
    )

    assert len(result.history) == len(loader) == 1
    assert sorted(result.replay.seen_indices) == [0, 2]


def test_mir_encoder_preserves_source_drop_last():
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 0])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")

    result = encoder.encode(
        dataloader=loader,
        set_name="source drop_last",
        n_replay_samples=1,
        loss_fn=code_length_loss,
        learning_rate=1e-3,
        seed=42,
        use_beta=False,
        use_ema=False,
    )

    assert len(result.history) == len(loader) == 1
    assert result.replay.seen_indices == [0, 1]


def test_mir_encoder_delegates_variable_batches_to_custom_replay():
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, 0])
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_sampler=VariableBatchSampler())
    replay = RecordingReplay(dataset, collate_fn=loader.collate_fn)
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")

    result = encoder.encode(
        dataloader=loader,
        set_name="custom replay batches",
        replay=replay,
        loss_fn=code_length_loss,
        learning_rate=1e-3,
        seed=42,
        use_beta=False,
        use_ema=False,
    )

    assert result.replay is replay
    assert len(result.history) == len(loader) == 2
    assert replay.updated_indices == [[2, 0], [1]]
    assert replay.sample_calls == len(loader)


def test_mir_encoder_rejects_unbatched_source_dataloader():
    dataset = TensorDataset(torch.ones(1, 2), torch.zeros(1, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=None)
    replay = RecordingReplay(dataset, collate_fn=loader.collate_fn)
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")

    with pytest.raises(ValueError, match="must use automatic or custom batching"):
        encoder.encode(
            dataloader=loader,
            set_name="unbatched source",
            replay=replay,
            loss_fn=code_length_loss,
            use_beta=False,
            use_ema=False,
        )


def test_mir_encoder_rejects_iterable_dataset():
    loader = DataLoader(SimpleIterableDataset(), batch_size=1)
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")

    with pytest.raises(TypeError, match="map-style dataset"):
        encoder.encode(
            dataloader=loader,
            set_name="iterable source",
            n_replay_samples=1,
            loss_fn=code_length_loss,
            use_beta=False,
            use_ema=False,
        )


def test_mir_calculate_code_length_passes_whole_batch_to_model_and_loss():
    model = BatchLinear()
    encoder = MIREncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")
    batch = next(iter(make_loader()))
    seen = {}

    def loss_fn(batch_arg, output):
        seen["batch"] = batch_arg
        seen["output"] = output
        return code_length_loss(batch_arg, output)

    state = EncoderState(
        model=model,
        optim=None,
        beta=None,
        beta_optim=None,
        ema_params=None,
        trained_params=None,
        loss_fn=loss_fn,
    )

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        state,
        batch,
        use_device_handling=False,
    )

    assert model.last_batch is batch
    assert seen["batch"] is batch
    assert seen["output"] is output
    assert returned_batch is batch
    assert code_lengths.shape == batch[1].shape


def test_loss_fn_is_required():
    encoder = BlockEncoder(ModelClass(BatchLinear, device="cpu"), device="cpu")
    loader = make_loader()

    try:
        encoder.encode([loader], [loader], "missing loss", seed=42)
    except ValueError as exc:
        assert "loss_fn" in str(exc)
    else:
        raise AssertionError("Expected loss_fn requirement to raise ValueError")
