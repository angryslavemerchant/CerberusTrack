import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def _dw_sep_block(c):
    return nn.Sequential(
        nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, c * 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
    )


class CerberusSiamese(nn.Module):
    """
    Siamese tracker.
      Template: 128x128 -> backbone -> (b, 96,  8,  8)  [K/V]
      Search:   256x256 -> backbone -> (b, 96, 16, 16)  [Q]
      xattn cross-attention -> heatmap (b, 1, 16, 16)

    embed_dim  : MobileNetV2 output channels at features[:14]  (96)
    num_channel: internal channel width after chan_proj / post_dw
    """

    def __init__(self, embed_dim=96, num_channel=64):
        super().__init__()

        # Shared backbone - weight-sharing is the defining property of Siamese nets.
        # MobileNetV2 1.0x, cut at features[:14] -> stride-16, 96ch.
        mv2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(mv2.features[:14]))

        # xattn: exact structure from mha_reference.py
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.chan_proj = nn.Conv2d(embed_dim, num_channel, kernel_size=1)
        self.post_dw = nn.Sequential(
            _dw_sep_block(num_channel),
            _dw_sep_block(num_channel),
        )

        # Heatmap head: dw-sep blocks for capacity, then 1x1 to score map (raw logits)
        self.heatmap_head = nn.Sequential(
            _dw_sep_block(num_channel),
            _dw_sep_block(num_channel),
            nn.Conv2d(num_channel, 1, kernel_size=1),
        )

    def forward(self, z, x):
        """
        z : template crop  (b, 3, 128, 128)
        x : search region  (b, 3, 256, 256)
        returns heatmap    (b,  1,  16,  16)  - raw logits, use BCEWithLogitsLoss
        """
        z_feat = self.backbone(z)  # (b, 96,  8,  8)
        x_feat = self.backbone(x)  # (b, 96, 16, 16)

        b = x_feat.size(0)
        hx, wx = x_feat.size(2), x_feat.size(3)

        # xattn - flatten(2) collapses spatial only, permute to sequence format
        x_seq = x_feat.flatten(2).permute(0, 2, 1)  # (b, 256, 96)  Q from search
        z_seq = z_feat.flatten(2).permute(0, 2, 1)  # (b,  64, 96)  K/V from template

        mha_out = self.mha(
            self.q_proj(x_seq),   # Q: (b, 256, 96)
            self.k_proj(z_seq),   # K: (b,  64, 96)
            self.v_proj(z_seq),   # V: (b,  64, 96)
        )[0]                      # (b, 256, 96)

        # permute -> reshape back to spatial - same pattern as Hailo reference
        corr = mha_out.permute(0, 2, 1).reshape(b, -1, hx, wx)  # (b, 96, 16, 16)
        corr = self.chan_proj(corr)                               # (b, 64, 16, 16)
        corr = self.post_dw(corr)                                 # (b, 64, 16, 16)

        return self.heatmap_head(corr)                            # (b,  1, 16, 16)
