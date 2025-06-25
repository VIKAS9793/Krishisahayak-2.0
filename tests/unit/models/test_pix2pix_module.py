import pytest
import torch
from unittest.mock import MagicMock

from krishi_sahayak.models.pix2pix import Pix2PixGAN, Pix2PixConfig

@pytest.fixture
def pix2pix_config() -> Pix2PixConfig:
    """Provides a default Pix2PixConfig for testing."""
    # Use smaller feature maps for faster testing
    config = Pix2PixConfig()
    config.generator.features = 8
    config.discriminator.features = 8
    return config

@pytest.fixture
def pix2pix_model(pix2pix_config: Pix2PixConfig) -> Pix2PixGAN:
    """Provides an instance of the Pix2PixGAN module."""
    return Pix2PixGAN(config=pix2pix_config)

@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    """Provides a dummy data batch."""
    return {
        "image": torch.randn(2, 3, 64, 64),    # input
        "ms_image": torch.randn(2, 1, 64, 64), # target
    }

class TestPix2PixGAN:
    """Tests the Pix2PixGAN LightningModule."""

    def test_instantiation(self, pix2pix_model: Pix2PixGAN):
        """Verify the model can be instantiated correctly."""
        assert isinstance(pix2pix_model, pl.LightningModule)
        assert hasattr(pix2pix_model, "generator")
        assert hasattr(pix2pix_model, "discriminator")

    def test_forward_pass(self, pix2pix_model: Pix2PixGAN, dummy_batch: Dict[str, torch.Tensor]):
        """Verify the forward pass calls the generator and returns correct shape."""
        input_tensor = dummy_batch["image"]
        output = pix2pix_model(input_tensor)
        
        expected_channels = pix2pix_model.config.generator.out_channels
        assert output.shape == (2, expected_channels, 64, 64)

    def test_training_step_generator(self, pix2pix_model: Pix2PixGAN, dummy_batch: Dict[str, torch.Tensor]):
        """Test the generator training step."""
        pix2pix_model.log_dict = MagicMock()
        loss = pix2pix_model.training_step(batch=dummy_batch, batch_idx=0, optimizer_idx=0)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        pix2pix_model.log_dict.assert_called_once()
        assert "g_loss" in pix2pix_model.log_dict.call_args[0][0]

    def test_training_step_discriminator(self, pix2pix_model: Pix2PixGAN, dummy_batch: Dict[str, torch.Tensor]):
        """Test the discriminator training step."""
        pix2pix_model.log = MagicMock()
        loss = pix2pix_model.training_step(batch=dummy_batch, batch_idx=0, optimizer_idx=1)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        pix2pix_model.log.assert_called_once_with('d_loss', loss, prog_bar=True)

    def test_validation_step(self, pix2pix_model: Pix2PixGAN, dummy_batch: Dict[str, torch.Tensor]):
        """Test the validation step logic."""
        pix2pix_model.log = MagicMock()
        pix2pix_model._log_image_grid = MagicMock()
        
        # Test first batch, where images should be logged
        pix2pix_model.validation_step(batch=dummy_batch, batch_idx=0)
        pix2pix_model.log.assert_called_with('val_pixel_loss', unittest.mock.ANY, on_step=False, on_epoch=True, prog_bar=True)
        pix2pix_model._log_image_grid.assert_called_once()
        
        # Test subsequent batch, images should not be logged
        pix2pix_model.log.reset_mock()
        pix2pix_model._log_image_grid.reset_mock()
        pix2pix_model.validation_step(batch=dummy_batch, batch_idx=1)
        pix2pix_model.log.assert_called_once()
        pix2pix_model._log_image_grid.assert_not_called()

    def test_configure_optimizers(self, pix2pix_model: Pix2PixGAN):
        """Verify that two optimizers and two schedulers are configured."""
        optimizers = pix2pix_model.configure_optimizers()
        assert isinstance(optimizers, tuple)
        assert len(optimizers) == 2
        
        opt_g_config, opt_d_config = optimizers
        assert "optimizer" in opt_g_config
        assert "lr_scheduler" in opt_g_config
        assert isinstance(opt_g_config["optimizer"], torch.optim.Adam)
        assert isinstance(opt_g_config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
