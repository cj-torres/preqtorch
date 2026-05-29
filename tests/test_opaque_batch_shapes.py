from dataclasses import dataclass

import torch
import torch.nn.functional as F

from preqtorch import BlockEncoder, MIREncoder, ModelClass
from preqtorch.encoders import EncoderState


class TupleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]))
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        features, _ = batch
        return self.linear(features)


class TeacherForcingSequenceModel(torch.nn.Module):
    def __init__(self, vocab_size=4, hidden_size=5):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.input_projection = torch.nn.Linear(2, hidden_size)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)
        self.last_teacher_tokens = None
        self.last_teacher_mask = None

    def forward(self, batch):
        source = batch["source"]
        decoder_inputs = batch["decoder_inputs"]
        teacher_mask = batch["teacher_forcing_mask"]
        teacher_tokens = decoder_inputs.masked_fill(~teacher_mask, 0)
        self.last_teacher_tokens = teacher_tokens
        self.last_teacher_mask = teacher_mask

        source_context = self.input_projection(source).unsqueeze(1)
        teacher_context = self.token_embedding(teacher_tokens)
        return self.output_projection(source_context + teacher_context)


class NestedRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.bias = torch.nn.Parameter(torch.tensor(-1.0))
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        features = batch[0]["features"]
        return features * self.scale + self.bias


@dataclass
class BinaryBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    sample_weights: torch.Tensor


class DataclassBinaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.5], [-0.5]]))
        self.bias = torch.nn.Parameter(torch.tensor([0.25]))
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        return batch.inputs @ self.weight + self.bias


def _block_encoder_for(model_type):
    return BlockEncoder(ModelClass(model_type, device="cpu"), device="cpu")


def _mir_encoder_for(model_type):
    return MIREncoder(ModelClass(model_type, device="cpu"), device="cpu")


def test_tuple_batch_with_cross_entropy_loss():
    model = TupleClassifier()
    encoder = _block_encoder_for(TupleClassifier)
    features = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 2.0]])
    targets = torch.tensor([1, 0])
    batch = (features, targets)

    def loss_fn(batch, output):
        _, labels = batch
        return F.cross_entropy(output, labels, reduction="none")

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        model,
        batch,
        loss_fn=loss_fn,
        use_device_handling=False,
    )

    assert returned_batch is batch
    assert model.last_batch is batch
    assert torch.equal(output, model.linear(features))
    assert torch.allclose(code_lengths, F.cross_entropy(output, targets, reduction="none"))


def test_dict_sequence_batch_with_teacher_forcing_and_masked_token_loss():
    model = TeacherForcingSequenceModel()
    encoder = _block_encoder_for(TeacherForcingSequenceModel)
    batch = {
        "source": torch.tensor([[0.5, -1.0], [1.0, 0.25]]),
        "decoder_inputs": torch.tensor([[1, 2, 3], [3, 2, 1]]),
        "teacher_forcing_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "targets": torch.tensor([[2, 3, 0], [1, 0, 0]]),
        "loss_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }

    def teacher_forced_loss(batch, output):
        loss_mask = batch["loss_mask"]
        targets = batch["targets"]
        return F.cross_entropy(output[loss_mask], targets[loss_mask], reduction="none")

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        model,
        batch,
        loss_fn=teacher_forced_loss,
        use_device_handling=False,
    )

    expected_teacher_tokens = batch["decoder_inputs"].masked_fill(~batch["teacher_forcing_mask"], 0)
    assert returned_batch is batch
    assert torch.equal(model.last_teacher_mask, batch["teacher_forcing_mask"])
    assert torch.equal(model.last_teacher_tokens, expected_teacher_tokens)
    assert output.shape == (2, 3, 4)
    assert code_lengths.shape == (batch["loss_mask"].sum().item(),)
    assert torch.allclose(
        code_lengths,
        F.cross_entropy(output[batch["loss_mask"]], batch["targets"][batch["loss_mask"]], reduction="none"),
    )


def test_nested_list_and_dict_batch_with_weighted_mse_loss_on_mir_encoder():
    model = NestedRegressionModel()
    encoder = _mir_encoder_for(NestedRegressionModel)
    batch = [
        {
            "features": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "metadata": {"source": "synthetic"},
        },
        {
            "targets": torch.tensor([[1.5, 2.0], [5.0, 7.0]]),
            "weights": torch.tensor([[1.0, 0.5], [0.25, 2.0]]),
        },
    ]

    def weighted_mse_loss(batch, output):
        targets = batch[1]["targets"]
        weights = batch[1]["weights"]
        return ((output - targets) ** 2 * weights).flatten()

    state = EncoderState(
        model=model,
        optim=None,
        beta=None,
        beta_optim=None,
        ema_params=None,
        trained_params=None,
        loss_fn=weighted_mse_loss,
    )

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        state,
        batch,
        use_device_handling=False,
    )

    expected_output = batch[0]["features"] * model.scale + model.bias
    expected_loss = ((expected_output - batch[1]["targets"]) ** 2 * batch[1]["weights"]).flatten()
    assert returned_batch is batch
    assert model.last_batch is batch
    assert torch.allclose(output, expected_output)
    assert torch.allclose(code_lengths, expected_loss)


def test_dataclass_batch_with_weighted_binary_loss_when_device_handling_is_user_owned():
    model = DataclassBinaryModel()
    encoder = _block_encoder_for(DataclassBinaryModel)
    batch = BinaryBatch(
        inputs=torch.tensor([[1.0, 2.0], [0.5, -1.0], [-1.0, 1.0]]),
        labels=torch.tensor([[1.0], [0.0], [1.0]]),
        sample_weights=torch.tensor([[1.0], [0.5], [2.0]]),
    )

    def weighted_binary_loss(batch, output):
        unweighted = F.binary_cross_entropy_with_logits(output, batch.labels, reduction="none")
        return (unweighted * batch.sample_weights).squeeze(-1)

    code_lengths, returned_batch, output = encoder.calculate_code_length(
        model,
        batch,
        loss_fn=weighted_binary_loss,
        use_device_handling=False,
    )

    expected = (
        F.binary_cross_entropy_with_logits(output, batch.labels, reduction="none") * batch.sample_weights
    ).squeeze(-1)
    assert returned_batch is batch
    assert model.last_batch is batch
    assert code_lengths.shape == (3,)
    assert torch.allclose(code_lengths, expected)
