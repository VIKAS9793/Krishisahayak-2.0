import pytest
import torch

from krishi_sahayak.models.gan import (
    GeneratorConfig,
    DiscriminatorConfig,
    EnhancedGenerator,
    EnhancedDiscriminator,
    SelfAttention,
)

class TestGANComponents:
    """Unit tests for individual GAN components and full architectures."""

    def test_self_attention_shape(self):
        """Verify that the SelfAttention layer preserves tensor shape."""
        batch_size, channels, height, width = 2, 64, 16, 16
        attention = SelfAttention(in_channels=channels)
        dummy_input = torch.randn(batch_size, channels, height, width)
        output = attention(dummy_input)
        assert output.shape == dummy_input.shape

    def test_generator_instantiation_and_shape(self):
        """
        Verify the Generator can be instantiated and produces an output
        tensor of the correct shape.
        """
        batch_size, height, width = 2, 256, 256
        config = GeneratorConfig(in_channels=3, out_channels=1, features=16)
        generator = EnhancedGenerator(config)
        
        dummy_input = torch.randn(batch_size, config.in_channels, height, width)
        output = generator(dummy_input)
        
        expected_shape = (batch_size, config.out_channels, height, width)
        assert output.shape == expected_shape

    def test_generator_no_attention(self):
        """Verify the Generator can be instantiated without the attention layer."""
        config = GeneratorConfig(features=16, use_attention=False)
        generator = EnhancedGenerator(config)
        assert not isinstance(generator.attention, SelfAttention)
        assert isinstance(generator.attention, torch.nn.Identity)

    def test_discriminator_instantiation_and_shape(self):
        """

        Verify the Discriminator can be instantiated and produces a patch
        output of the correct shape.
        """
        batch_size, height, width = 2, 256, 256
        # Discriminator input is concatenation of real and fake image
        config = DiscriminatorConfig(in_channels=3 + 1, features=16)
        discriminator = EnhancedDiscriminator(config)
        
        # img_A is the input (e.g., satellite image), img_B is the target (e.g., mask)
        img_A = torch.randn(batch_size, 3, height, width)
        img_B = torch.randn(batch_size, 1, height, width)
        
        output = discriminator(img_A, img_B)
        
        # PatchGAN output is smaller than the input, not a single scalar
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1
        assert output.shape[2] > 1 and output.shape[3] > 1

    def test_spectral_norm_application(self):
        """
        Verify that spectral normalization is applied to layers when configured.
        """
        config = DiscriminatorConfig(use_spectral_norm=True, features=8)
        discriminator = EnhancedDiscriminator(config)
        
        # Check if the first convolutional layer has the spectral_norm wrapper
        first_conv = discriminator.model[0].model[0]
        assert hasattr(first_conv, 'weight_orig')
        
        # Now check the opposite
        config_no_sn = DiscriminatorConfig(use_spectral_norm=False, features=8)
        discriminator_no_sn = EnhancedDiscriminator(config_no_sn)
        first_conv_no_sn = discriminator_no_sn.model[0].model[0]
        assert not hasattr(first_conv_no_sn, 'weight_orig')
