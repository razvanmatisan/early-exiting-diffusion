import torch

from models.early_exit import AttentionProbe, EarlyExitUViT, OutputHead
from models.uvit import UViT

celeba_config = dict(
    img_size=64,
    patch_size=4,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    num_classes=-1,
)

cifar10_config = dict(
    img_size=32,
    patch_size=2,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=False,
    mlp_time_embed=False,
    num_classes=-1,
)

batch_size = 8
num_channels = 3
height = 32
width = 32
x = torch.zeros((batch_size, num_channels, height, width))
t = torch.ones(batch_size)


def test_output_head():
    output_head = OutputHead(embed_dim=512, patch_dim=2**2 * 3, in_chans=3, conv=True)

    x = torch.zeros((16, 257, 512))

    y = output_head(x)
    assert y.shape == (16, 3, 32, 32)


def test_attention_probe():
    attention_probe = AttentionProbe(embed_dim=512, num_heads=1)

    x = torch.zeros((16, 257, 512))
    y = attention_probe(x)

    assert y.shape == (16,)


def test_backward():
    model = EarlyExitUViT(UViT(**cifar10_config))
    y, classifier_outputs, outputs = model(x, t)

    assert y.shape == x.shape
    assert len(outputs) == len(classifier_outputs) == 13

    fake_loss = torch.sum(y)
    fake_loss.backward()


def test_inference_no_early_exit():
    model = EarlyExitUViT(UViT(**cifar10_config), exit_threshold=-torch.inf)
    model.eval()
    with torch.inference_mode():
        y, classifier_outputs = model(x, t)

    assert len(classifier_outputs) == 13
    assert y.shape == (batch_size, num_channels, height, width)


def test_inference_exit_first():
    model = EarlyExitUViT(UViT(**cifar10_config), exit_threshold=torch.inf)
    model.eval()
    with torch.inference_mode():
        y, classifier_outputs = model(x, t)

    assert y.shape == (batch_size, num_channels, height, width)
    assert len(classifier_outputs) == 1
    assert all(classifier_outputs[0] < model.exit_threshold)