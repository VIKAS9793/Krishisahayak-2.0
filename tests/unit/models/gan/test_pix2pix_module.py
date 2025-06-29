import pytest
import torch
from typing import Dict
from krishi_sahayak.models.gan.pix2pix import Pix2PixGAN, Pix2PixConfig

@pytest.fixture
def pix2pix_config() -> Pix2PixConfig:
    return Pix2PixConfig()

@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    return {
        "image": torch.randn(2, 3, 256, 256),
        "ms_image": torch.randn(2, 1, 256, 256),
    }

def test_pix2pix_instantiation(pix2pix_config: Pix2PixConfig):
    model = Pix2PixGAN(config=pix2pix_config)
    assert model is not None
    assert isinstance(model.generator, torch.nn.Module)
    assert isinstance(model.discriminator, torch.nn.Module)

def test_pix2pix_training_step(pix2pix_config: Pix2PixConfig, dummy_batch: Dict[str, torch.Tensor]):
    model = Pix2PixGAN(config=pix2pix_config)
    # Test generator step
    g_loss = model.training_step(dummy_batch, 0, 0)
    assert g_loss is not None and g_loss > 0
    # Test discriminator step
    d_loss = model.training_step(dummy_batch, 0, 1)
    assert d_loss is not None and d_loss > 0
