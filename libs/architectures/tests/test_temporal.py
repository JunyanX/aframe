import torch
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.nn.resnet.resnet_1d import ResNet1D

from architectures.supervised import SupervisedTimeDomainResNet


def make_model(**kwargs):
    return SupervisedTimeDomainResNet(
        num_ifos=2,
        sample_rate=2048,
        kernel_length=1.5,
        layers=[1, 1, 1, 1],
        norm_layer=GroupNorm1DGetter(groups=16),
        **kwargs,
    )


def test_average_head_is_unchanged():
    original = ResNet1D(
        2,
        layers=[1, 1, 1, 1],
        classes=1,
        norm_layer=GroupNorm1DGetter(groups=16),
    ).eval()
    model = make_model().eval()
    model.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 2, 3072)
    torch.testing.assert_close(model(x), original(x))


def test_temporal_head_output_and_trace():
    model = make_model(head_type="temporal").eval()
    x = torch.randn(4, 2, 3072)
    assert model(x).shape == (4, 1)
    traced = torch.jit.trace(model, x)
    torch.testing.assert_close(traced(x), model(x))
