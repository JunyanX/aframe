import torch
from ml4gw.nn.norm import GroupNorm1DGetter

from architectures.supervised import SupervisedTimeDomainResNet


def make_model(**kwargs):
    return SupervisedTimeDomainResNet(
        num_ifos=2,
        sample_rate=2048,
        kernel_length=1.5,
        layers=[1, 1, 1, 1],
        norm_layer=GroupNorm1DGetter(groups=16),
        head_type="temporal",
        **kwargs,
    )


def test_attention_pooling_starts_as_average_pooling():
    average = make_model().eval()
    attention = make_model(temporal_pooling="attention").eval()
    result = attention.load_state_dict(average.state_dict(), strict=False)
    assert result.missing_keys == ["avgpool.pool.score.weight"]
    assert not result.unexpected_keys
    x = torch.randn(2, 2, 3072)
    torch.testing.assert_close(attention(x), average(x))


def test_attention_scores_receive_gradient():
    model = make_model(temporal_pooling="attention")
    model(torch.randn(4, 2, 3072)).sum().backward()
    assert model.avgpool.pool.score.weight.grad.abs().sum() > 0


def test_temporal_attention_output_and_trace():
    model = make_model(temporal_pooling="attention").eval()
    x = torch.randn(4, 2, 3072)
    assert model(x).shape == (4, 1)
    traced = torch.jit.trace(model, x)
    torch.testing.assert_close(traced(x), model(x))
