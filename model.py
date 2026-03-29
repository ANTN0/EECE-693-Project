"""
model.py - All 4 model variants in one file.

    Model 1: U-Net, grayscale -> RGB, L1 loss (baseline)
    Model 2: U-Net, L channel -> ab channels, L1+L2 loss (Lab color space)
    Model 3: Model 2 + VGG perceptual loss (sharper colors)
    Model 4: Model 3 + reference-guided cross-attention (our contribution)

The core U-Net architecture is shared. Each model just changes:
    - Input/output channels
    - Whether cross-attention is used (Model 4 only)
    - Loss function (handled in train.py, not here)
"""

import torch
import torch.nn as nn

import config


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU. Basic U-Net building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool for downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    """Upsample + skip connection + ConvBlock + optional Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.dropout(x)


# =============================================================================
# CROSS-ATTENTION (Model 4 only)
# =============================================================================

class CrossAttention(nn.Module):
    """
    Simple cross-attention: target features query reference features
    to pull relevant color information.

    target_features:    [B, C, H, W]  (from the decoder)
    reference_features: [B, C, H, W]  (from the reference encoder)
    output:             [B, C, H, W]  (target enriched with reference colors)
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, target, reference):
        B, C, H, W = target.shape

        # Flatten spatial dims: [B, C, H*W]
        q = self.query(target).view(B, C, -1)
        k = self.key(reference).view(B, C, -1)
        v = self.value(reference).view(B, C, -1)

        # Attention: softmax(Q^T K / sqrt(C)) * V
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))  # [B, C, HW]

        return out.view(B, C, H, W) + target  # residual connection


class DecoderBlockWithCrossAttn(nn.Module):
    """Decoder block that also attends to reference features (Model 4)."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)
        self.cross_attn = CrossAttention(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip, ref_features=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if ref_features is not None:
            x = self.cross_attn(x, ref_features)
        return self.dropout(x)


# =============================================================================
# REFERENCE ENCODER (Model 4 only)
# =============================================================================

class ReferenceEncoder(nn.Module):
    """
    Encodes K reference color images into multi-scale feature maps.
    Uses the same encoder structure as the main U-Net but with 3-channel input (RGB).
    Features from all K references are averaged at each scale.
    """
    def __init__(self, encoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = config.ENCODER_CHANNELS

        self.enc1 = EncoderBlock(3, encoder_channels[0])
        self.enc2 = EncoderBlock(encoder_channels[0], encoder_channels[1])
        self.enc3 = EncoderBlock(encoder_channels[1], encoder_channels[2])
        self.enc4 = EncoderBlock(encoder_channels[2], encoder_channels[3])

    def forward(self, refs):
        """
        refs: [B, K, 3, H, W] — K reference images per batch item.
        Returns: list of 4 feature maps (one per encoder level), each [B, C, H', W'].
        """
        B, K, C, H, W = refs.shape

        # Process all references together by reshaping
        x = refs.view(B * K, C, H, W)

        f1, x = self.enc1(x)
        f2, x = self.enc2(x)
        f3, x = self.enc3(x)
        f4, _ = self.enc4(x)

        # Average features across the K references for each batch item
        def avg_refs(feat):
            _, c, h, w = feat.shape
            return feat.view(B, K, c, h, w).mean(dim=1)  # [B, C, H, W]

        return [avg_refs(f1), avg_refs(f2), avg_refs(f3), avg_refs(f4)]


# =============================================================================
# UNIFIED MODEL
# =============================================================================

class RestorationModel(nn.Module):
    """
    Unified model for all 4 variants. The model_variant parameter controls behavior:

        Model 1: 1 ch in -> 3 ch out (grayscale -> RGB)
        Model 2: 1 ch in -> 2 ch out (L -> ab), same architecture
        Model 3: same as Model 2 (perceptual loss is added in train.py)
        Model 4: same as Model 3 + reference branch with cross-attention
    """

    def __init__(self, model_variant=None):
        super().__init__()
        self.model_variant = model_variant or config.MODEL_VARIANT

        enc_ch = config.ENCODER_CHANNELS
        bneck = config.BOTTLENECK_CHANNELS
        drop = config.DROPOUT_RATE

        # Input: always 1 channel (grayscale or L channel)
        in_ch = 1

        # Output: 3 for RGB (model 1), 2 for ab channels (models 2-4)
        if self.model_variant == 1:
            out_ch = 3
        else:
            out_ch = 2

        # --- Encoder ---
        self.enc1 = EncoderBlock(in_ch, enc_ch[0])
        self.enc2 = EncoderBlock(enc_ch[0], enc_ch[1])
        self.enc3 = EncoderBlock(enc_ch[1], enc_ch[2])
        self.enc4 = EncoderBlock(enc_ch[2], enc_ch[3])

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(enc_ch[3], bneck)

        # --- Decoder ---
        if self.model_variant == 4:
            # Model 4: decoder blocks with cross-attention to reference features
            self.dec4 = DecoderBlockWithCrossAttn(bneck, enc_ch[3], drop)
            self.dec3 = DecoderBlockWithCrossAttn(enc_ch[3], enc_ch[2], drop)
            self.dec2 = DecoderBlockWithCrossAttn(enc_ch[2], enc_ch[1], drop)
            self.dec1 = DecoderBlockWithCrossAttn(enc_ch[1], enc_ch[0], drop)
            # Reference encoder
            self.ref_encoder = ReferenceEncoder(enc_ch)
        else:
            # Models 1-3: standard decoder
            self.dec4 = DecoderBlock(bneck, enc_ch[3], drop)
            self.dec3 = DecoderBlock(enc_ch[3], enc_ch[2], drop)
            self.dec2 = DecoderBlock(enc_ch[2], enc_ch[1], drop)
            self.dec1 = DecoderBlock(enc_ch[1], enc_ch[0], drop)

        # --- Output heads ---
        self.color_head = nn.Sequential(
            nn.Conv2d(enc_ch[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        # Damage classification head (shared across all models)
        self.damage_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bneck, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, config.NUM_DAMAGE_TYPES),
        )

    def forward(self, x, refs=None):
        """
        x:    [B, 1, H, W]  — grayscale or L channel input
        refs: [B, K, 3, H, W] — reference images (Model 4 only, ignored otherwise)

        Returns:
            color_out:     [B, out_ch, H, W]  — RGB (model 1) or ab (models 2-4)
            damage_logits: [B, num_damage_types]
        """
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        bn = self.bottleneck(x)

        # Decoder
        if self.model_variant == 4 and refs is not None:
            # Get reference features at each scale
            ref_feats = self.ref_encoder(refs)  # [f1, f2, f3, f4]
            x = self.dec4(bn, s4, ref_feats[3])
            x = self.dec3(x, s3, ref_feats[2])
            x = self.dec2(x, s2, ref_feats[1])
            x = self.dec1(x, s1, ref_feats[0])
        else:
            x = self.dec4(bn, s4)
            x = self.dec3(x, s3)
            x = self.dec2(x, s2)
            x = self.dec1(x, s1)

        color_out = self.color_head(x)
        damage_logits = self.damage_head(bn)

        return color_out, damage_logits


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    for variant in [1, 2, 3, 4]:
        print(f"\n--- Model {variant} ---")
        model = RestorationModel(model_variant=variant)
        dummy = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)

        if variant == 4:
            refs = torch.randn(2, config.NUM_REFERENCES, 3,
                               config.IMAGE_SIZE, config.IMAGE_SIZE)
            out, dmg = model(dummy, refs)
        else:
            out, dmg = model(dummy)

        print(f"  Input:  {dummy.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Damage: {dmg.shape}")
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")
