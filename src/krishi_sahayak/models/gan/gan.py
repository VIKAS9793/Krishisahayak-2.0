# src/krishi_sahayak/models/gan/gan.py
"""Advanced GAN Models for Image-to-Image Translation"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from pydantic import BaseModel, Field
from torch.nn.utils import spectral_norm

class GeneratorConfig(BaseModel):
    in_channels: int=3; out_channels: int=1; features: int=64; leaky_relu_slope: float=0.2; use_attention: bool=True; attention_projection_ratio: int=8; use_spectral_norm: bool=False; dropout_rate: float=0.5

class DiscriminatorConfig(BaseModel):
    in_channels: int=4; features: int=64; leaky_relu_slope: float=0.2; use_spectral_norm: bool=True; use_attention: bool=False; attention_projection_ratio: int=8

class GANConfig(BaseModel):
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig); discriminator: DiscriminatorConfig = Field(default_factory=DiscriminatorConfig)

def weights_init_xavier(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.xavier_normal_(m.weight)

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, projection_ratio: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.projection_dim = max(1, in_channels // projection_ratio)  # Ensure at least 1 dimension
        
        self.query = spectral_norm(nn.Conv1d(in_channels, self.projection_dim, 1))
        self.key = spectral_norm(nn.Conv1d(in_channels, self.projection_dim, 1))
        self.value = spectral_norm(nn.Conv1d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1)  # Shape: [B, C, H*W]
        
        # Compute query, key, value
        q = self.query(x_flat).permute(0, 2, 1)  # [B, H*W, projection_dim]
        k = self.key(x_flat)  # [B, projection_dim, H*W]
        v = self.value(x_flat)  # [B, C, H*W]
        
        # Compute attention scores
        attn = F.softmax(torch.bmm(q, k), dim=-1)  # [B, H*W, H*W]
        
        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)  # Reshape back to original spatial dimensions
        
        return self.gamma * out + x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, use_norm=True, use_sn=False, leaky_relu_slope=0.2):
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm)
        layers = [spectral_norm(conv) if use_sn else conv]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0, use_sn=False):
        super().__init__()
        conv_t = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        layers = [spectral_norm(conv_t) if use_sn else conv_t, 
                 nn.InstanceNorm2d(out_c), 
                 nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat([x, skip_input], 1)

class EnhancedGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        f = config.features
        dr = config.dropout_rate
        sn = config.use_spectral_norm
        slope = config.leaky_relu_slope
        
        # Downsampling path
        self.down1 = DownBlock(config.in_channels, f, use_norm=False, use_sn=sn, leaky_relu_slope=slope)
        self.down2 = DownBlock(f, f*2, use_sn=sn, leaky_relu_slope=slope)
        self.down3 = DownBlock(f*2, f*4, use_sn=sn, leaky_relu_slope=slope)
        self.down4 = DownBlock(f*4, f*8, use_sn=sn, leaky_relu_slope=slope)
        self.down5 = DownBlock(f*8, f*8, use_sn=sn, leaky_relu_slope=slope)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f*8, f*8, 4, 2, 1),
            nn.ReLU(True)
        )
        
        # Upsampling path
        self.up1 = UpBlock(f*8, f*8, dropout=dr, use_sn=sn)
        self.up2 = UpBlock(f*16, f*8, dropout=dr, use_sn=sn)
        
        # Attention layer with channel adjustment
        if config.use_attention:
            # Add a 1x1 conv to reduce channels before attention
            self.attention_conv = nn.Conv2d(f*16, f*8, kernel_size=1)
            self.attention = SelfAttention(f*8, config.attention_projection_ratio)
            # 1x1 conv to restore channels after attention
            self.attention_restore = nn.Conv2d(f*8, f*16, kernel_size=1)
        else:
            self.attention = nn.Identity()
            self.attention_conv = nn.Identity()
            self.attention_restore = nn.Identity()
        
        # Continue upsampling
        self.up3 = UpBlock(f*16, f*4, use_sn=sn)
        self.up4 = UpBlock(f*8, f*2, use_sn=sn)
        self.up5 = UpBlock(f*4, f, use_sn=sn)
        
        # Final output
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(f*2, config.out_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)      # [B, f, 128, 128]
        d2 = self.down2(d1)     # [B, f*2, 64, 64]
        d3 = self.down3(d2)     # [B, f*4, 32, 32]
        d4 = self.down4(d3)     # [B, f*8, 16, 16]
        d5 = self.down5(d4)     # [B, f*8, 8, 8]
        
        # Bottleneck
        b = self.bottleneck(d5)  # [B, f*8, 4, 4]
        
        # Decoder with skip connections
        u1 = self.up1(b, d5)    # [B, f*16, 8, 8]
        u2 = self.up2(u1, d4)   # [B, f*16, 16, 16]
        
        # Apply attention
        if isinstance(self.attention, nn.Identity):
            att = u2
        else:
            # Reduce channels, apply attention, then restore channels
            reduced = self.attention_conv(u2)
            att = self.attention(reduced)
            att = self.attention_restore(att)
            # Residual connection
            att = att + u2
        
        # Continue upsampling
        u3 = self.up3(att, d3)   # [B, f*8, 32, 32]
        u4 = self.up4(u3, d2)    # [B, f*4, 64, 64]
        u5 = self.up5(u4, d1)    # [B, f*2, 128, 128]
        
        # Final upsampling to original size
        return self.final_up(u5)  # [B, out_channels, 256, 256]

class EnhancedDiscriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__(); f,sn,slope=config.features,config.use_spectral_norm,config.leaky_relu_slope; layers=[
            DownBlock(config.in_channels,f,use_norm=False,use_sn=sn,leaky_relu_slope=slope),
            DownBlock(f,f*2,use_sn=sn,leaky_relu_slope=slope),
            DownBlock(f*2,f*4,use_sn=sn,leaky_relu_slope=slope),
            DownBlock(f*4,f*8,use_sn=sn,leaky_relu_slope=slope)]
        if config.use_attention: layers.append(SelfAttention(f*8,config.attention_projection_ratio))
        layers.append(nn.Conv2d(f*8,1,kernel_size=4,stride=1,padding=1)); self.model=nn.Sequential(*layers)
    def forward(self,img_A,img_B): return self.model(torch.cat([img_A,img_B],dim=1))

def create_enhanced_gan_models(config: GANConfig):
    """Factory function to create and initialize GAN models from a config object."""
    generator=EnhancedGenerator(config.generator); discriminator=EnhancedDiscriminator(config.discriminator)
    generator.apply(weights_init_xavier); discriminator.apply(weights_init_xavier); return generator,discriminator