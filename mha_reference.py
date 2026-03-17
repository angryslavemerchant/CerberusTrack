def __init__(self, num_channel, cat=False, CA=True, matrix=False, corr_type='mat',
             embed_dim=96, search_num=256):
    super(PWCA, self).__init__()
    self.cat = cat
    self.CA = CA
    self.corr_type = corr_type
    if self.CA:
        self.CA_layer = CAModule(channels=num_channel)
    if corr_type == 'mha':
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        # adapter: map MHA output (b, hz*wz, embed_dim) → (b, hz*wz, search_num)
        # then reshape to (b, num_channel, hx, wx) in forward
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, search_num),
            nn.ReLU(inplace=True),
            nn.Linear(search_num, search_num),
        )
    if corr_type == 'self_attn':
        # Exact Hailo example structure — self-attention on x
        # Q/K/V project from embed_dim → num_channel so output is (b, num_channel, hx, wx)
        self.q_proj = nn.Linear(embed_dim, num_channel)
        self.k_proj = nn.Linear(embed_dim, num_channel)
        self.v_proj = nn.Linear(embed_dim, num_channel)
        self.mha = nn.MultiheadAttention(embed_dim=num_channel, num_heads=1, batch_first=True)
    if corr_type == 'flat_attn':
        # Cross-attention: Q from z (template), K/V from x (search), both fully flattened.
        # z: (b, 96, 8, 8) → flatten → (b, 6144) → view (b, 64, 96)   — hz*wz tokens
        # x: (b, 96,16,16) → flatten → (b,24576) → view (b,256, 96)   — hx*wx tokens
        # MHA output: (b, 64, embed_dim) — one output token per template position
        # adapter maps per-token (b, 64, embed_dim) → (b, 64, search_num) → view (b, 64, 16, 16)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, search_num),
            nn.ReLU(inplace=True),
            nn.Linear(search_num, search_num),
        )
    if corr_type == 'xattn':
        # Cross-attention following the exact Hailo MHA example reshape pattern.
        # Key difference from all previous attempts: flatten(2) collapses spatial dims
        # only (keeping channels intact), then permute — both are transpose ops (supported).
        # The DFC is expected to fuse the full flatten(2)→permute→MHA→permute→reshape
        # sequence as a single MHA block, bypassing standalone reshape rules.
        #
        # Shape flow:
        #   x: (b, 96, 16, 16) → flatten(2) → (b, 96, 256) → permute(0,2,1) → (b, 256, 96) [Q]
        #   z: (b, 96,  8,  8) → flatten(2) → (b, 96,  64) → permute(0,2,1) → (b,  64, 96) [K/V]
        #   Q from x (search) so output seq len = 256, spatial shape = (16, 16)
        #   MHA output: (b, 256, embed_dim)
        #   permute(0,2,1) → (b, embed_dim, 256) → reshape(b, embed_dim, 16, 16)  ← same H,W as x ✓
        #   chan_proj: Conv2d(embed_dim, num_channel, 1) → (b, 64, 16, 16)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.chan_proj = nn.Conv2d(embed_dim, num_channel, kernel_size=1)

        # Two depthwise-separable blocks (DW + PW expand + PW contract) approximating
        # the transformer FFN pattern (two-layer MLP post-attention, 2× hidden expansion).
        # Each block: DW mixes spatially, PW expand/contract mixes channels — closest
        # Hailo-compatible equivalent to a Dense layer on the MHA output.
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

        self.post_dw = nn.Sequential(
            _dw_sep_block(num_channel),
            _dw_sep_block(num_channel),
        )
    if corr_type == 'single_token':
        # Single-token cross-attention — the key difference from flat_attn:
        # Instead of view(b, seq_len, c) (general reshape → unsupported by Hailo),
        # we flatten each branch to 1D (Conv-to-Dense: supported), project to embed_dim
        # via Linear, then unsqueeze(1) to get a (b, 1, embed_dim) sequence.
        # unsqueeze adds a dim of size 1 which is simpler than a general reshape and
        # may pass the DFC where multi-token views failed.
        #
        # Shape flow:
        #   z: (b, 96, 8, 8)   → flatten(1) → (b, 6144)  → z_proj → (b, embed_dim) → unsqueeze(1) → (b, 1, embed_dim)
        #   x: (b, 96, 16, 16) → flatten(1) → (b, 24576) → x_proj → (b, embed_dim) → unsqueeze(1) → (b, 1, embed_dim)
        #   MHA cross-attn Q=z_token, K/V=x_token → (b, 1, embed_dim)
        #   squeeze(1) → (b, embed_dim) → out_proj → (b, search_num=256)
        #   view(b, 1, hx, wx) — Dense-to-Conv: (b, N) → (b, 1, H, W) ✓
        #   spatial_conv: Conv2d(1, num_channel, 1) → (b, num_channel, hx, wx)
        z_flat_dim = embed_dim * num_channel  # 96 * 64 = 6144  (hz*wz = num_channel)
        x_flat_dim = embed_dim * search_num  # 96 * 256 = 24576
        self.z_proj = nn.Linear(z_flat_dim, embed_dim)
        self.x_proj = nn.Linear(x_flat_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, search_num)
        self.spatial_conv = nn.Conv2d(1, num_channel, kernel_size=1)


def forward(self, z, x):
    z11 = z[0]
    x11 = x[0]
    if self.corr_type == 'mha':
        b, c, hz, wz = z11.size()
        hx, wx = x11.size(2), x11.size(3)
        q = z11.view(b, hz * wz, c)  # (b, hz*wz, c)  — template positions as queries
        k = x11.view(b, hx * wx, c)  # (b, hx*wx, c)  — search positions as keys/values
        mha_out = self.mha(self.q_proj(q), self.k_proj(k), self.v_proj(k))[0]  # (b, hz*wz, embed_dim)
        corr = self.adapter(mha_out)  # (b, hz*wz, hx*wx)
        corr = corr.view(b, hz * wz, hx, wx)  # (b, num_channel, hx, wx)
    elif self.corr_type == 'self_attn':
        # Exact Hailo example structure
        b, c, hx, wx = x11.size()
        x_flat = x11.flatten(2).permute(0, 2, 1)  # (b, hx*wx, c)
        mha_out = self.mha(self.q_proj(x_flat), self.k_proj(x_flat), self.v_proj(x_flat))[0]  # (b, hx*wx, num_channel)
        corr = mha_out.permute(0, 2, 1).reshape(b, -1, hx, wx)  # (b, num_channel, hx, wx)
    elif self.corr_type == 'xattn':
        b, c, hz, wz = z11.size()
        hx, wx = x11.size(2), x11.size(3)
        # flatten(2): collapse spatial dims only — keeps channels, unlike flatten(1)
        # permute: transpose to sequence format — both ops match Hailo example exactly
        x_seq = x11.flatten(2).permute(0, 2, 1)  # (b, 256, 96) — Q from search
        z_seq = z11.flatten(2).permute(0, 2, 1)  # (b,  64, 96) — K/V from template
        mha_out = self.mha(
            self.q_proj(x_seq),  # Q: (b, 256, embed_dim)
            self.k_proj(z_seq),  # K: (b,  64, embed_dim)
            self.v_proj(z_seq),  # V: (b,  64, embed_dim)
        )[0]  # (b, 256, embed_dim)
        # reshape back following Hailo example: permute → reshape to same H,W as Q input
        corr = mha_out.permute(0, 2, 1).reshape(b, -1, hx, wx)  # (b, embed_dim, 16, 16)
        corr = self.chan_proj(corr)  # (b, num_channel, 16, 16)
        corr = self.post_dw(corr)  # (b, num_channel, 16, 16)
    elif self.corr_type == 'flat_attn':
        b, c, hz, wz = z11.size()
        hx, wx = x11.size(2), x11.size(3)
        q = z11.flatten(1).view(b, hz * wz, c)  # (b, 6144) → (b, 64, 96)
        k = x11.flatten(1).view(b, hx * wx, c)  # (b, 24576) → (b, 256, 96)
        mha_out = self.mha(self.q_proj(q), self.k_proj(k), self.v_proj(k))[0]  # (b, 64, 96)
        corr = self.adapter(mha_out)  # (b, 64, 256)
        corr = corr.view(b, hz * wz, hx, wx)  # (b, 64, 16, 16)
    elif self.corr_type == 'single_token':
        b, c, hz, wz = z11.size()
        hx, wx = x11.size(2), x11.size(3)
        # Conv-to-Dense flatten (supported by Hailo)
        z_flat = z11.flatten(1)  # (b, c*hz*wz)
        x_flat = x11.flatten(1)  # (b, c*hx*wx)
        # Project to embed_dim, then unsqueeze to single-token sequence
        z_token = self.z_proj(z_flat).unsqueeze(1)  # (b, 1, embed_dim)
        x_token = self.x_proj(x_flat).unsqueeze(1)  # (b, 1, embed_dim)
        # Cross-attention: Q from z, K/V from x
        mha_out = self.mha(z_token, x_token, x_token)[0]  # (b, 1, embed_dim)
        # squeeze → project → Dense-to-Conv reshape (b, N) → (b, 1, H, W)
        out = mha_out.squeeze(1)  # (b, embed_dim)
        out = self.out_proj(out)  # (b, search_num=256)
        out = out.view(b, 1, hx, wx)  # (b, 1, 16, 16) — Dense-to-Conv ✓
        corr = self.spatial_conv(out)  # (b, num_channel, 16, 16)
    elif self.corr_type == 'mat':
        corr = pixel_corr_mat(z11, x11)
    else:
        corr = pixel_corr(z11, x11)
    if self.CA:
        opt = self.CA_layer(corr)
        if self.cat:
            return torch.cat([opt, x11], dim=1)
        else:
            return opt
    else:
        return corr
