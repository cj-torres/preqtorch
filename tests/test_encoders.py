import torch
from torch.utils.data import DataLoader, TensorDataset

from preqtorch import BlockEncoder, EncoderResult, MIREncoder, ModelClass
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
    assert result.code_length.item() > 0
    assert len(result.history) == len(loader)
    assert result.replay is not None


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
