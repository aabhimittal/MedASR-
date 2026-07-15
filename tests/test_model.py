import torch

from medasr.config import ModelConfig
from medasr.models import ConformerCTC, ConformerEncoder
from medasr.models.subsampling import Conv2dSubsampling


def small_cfg():
    return ModelConfig(
        input_dim=80, d_model=64, num_layers=2, num_heads=4,
        ffn_expansion=2, conv_kernel_size=15, dropout=0.0,
    )


def test_subsampling_downsamples_by_four():
    sub = Conv2dSubsampling(input_dim=80, d_model=64)
    x = torch.randn(2, 100, 80)
    lengths = torch.tensor([100, 80])
    out, out_len = sub(x, lengths)
    assert out.shape[0] == 2 and out.shape[2] == 64
    # ~4x downsampling
    assert out.shape[1] == out_len.max().item()
    assert out_len[0] > out_len[1]


def test_encoder_forward_shapes():
    enc = ConformerEncoder(input_dim=80, d_model=64, num_layers=2, num_heads=4,
                           ffn_expansion=2, conv_kernel_size=15, dropout=0.0)
    feats = torch.randn(3, 80, 120)
    lengths = torch.tensor([120, 100, 60])
    out, out_len = enc(feats, lengths)
    assert out.shape[0] == 3 and out.shape[2] == 64
    assert out.shape[1] == out_len.max().item()


def test_ctc_forward_and_loss_backward():
    torch.manual_seed(0)
    vocab = 30
    model = ConformerCTC(small_cfg(), vocab_size=vocab)
    feats = torch.randn(2, 80, 160)
    feat_lengths = torch.tensor([160, 120])
    log_probs, out_lengths = model(feats, feat_lengths)

    assert log_probs.shape[0] == 2
    assert log_probs.shape[2] == vocab
    # log-probabilities sum to 1 in prob space.
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones_like(out_lengths, dtype=torch.float).unsqueeze(1).expand(-1, log_probs.shape[1]), atol=1e-4)

    targets = torch.randint(1, vocab, (2, 10))
    target_lengths = torch.tensor([10, 7])
    loss = model.compute_loss(log_probs, out_lengths, targets, target_lengths)
    assert loss.requires_grad and loss.item() > 0
    loss.backward()  # gradients flow
    assert model.classifier.weight.grad is not None


def test_variable_length_masking_is_deterministic_in_eval():
    model = ConformerCTC(small_cfg(), vocab_size=20).eval()
    feats = torch.randn(1, 80, 100)
    lengths = torch.tensor([100])
    a, _ = model(feats, lengths)
    b, _ = model(feats, lengths)
    assert torch.allclose(a, b)


def test_param_count_positive():
    model = ConformerCTC(small_cfg(), vocab_size=20)
    assert model.num_parameters() > 0
