from other_impls import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self, hidden_size, frequency_embedding_size=256, dtype=None, device=None
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs): #🔹 Add QRM quality vector
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)



        return t_emb
    
class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class QRMModulatorLatent(nn.Module):
    """
    Goal: produce q_t ∈ R^{out_dim} (e.g., 1536) that adjusts modulation based on the current latent.
    Inputs:
      - t:            [B] or [B,] diffusion time (float)
      - latent:       [B, 16, 64, 64] latent image at current step
      - y:            [B, 2048] pooled prompt embedding (same as your current 'y')
      - c_crossattn:  Optional[B, L, 4096] full text sequence (from SD3 encoders)
    Output:
      - q_t:          [B, out_dim] unit-direction * positive-magnitude
    """
    def __init__(self, out_dim=1536, time_bool=True, dropout=0.1,
                 latent_ch=16, token_hw=8, num_heads=8, use_text_seq=True,vision_dim=None):
        super().__init__()
        self.time_bool   = time_bool
        self.use_text_seq = use_text_seq
        D = out_dim
        H = num_heads

        # --- Embedders (match your existing widths so downstream stays plug-compatible) ---
        self.t_embedder = TimestepEmbedder(1536) if time_bool else None
        self.y_embedder = VectorEmbedder(2048, 1536)

        # --- Latent patchification: 16x64x64 -> KxK tokens (K=token_hw, e.g., 8 => 64 tokens) ---
        self.latent_proj = nn.Conv2d(latent_ch, D, kernel_size=1, bias=True)
        self.down        = nn.AdaptiveAvgPool2d((token_hw, token_hw))
        self.register_buffer("pos_embed", self._build_2d_sincos_pos_embed(D, token_hw), persistent=False)
        self.latent_norm = nn.LayerNorm(D)

        # --- KV tokens from pooled y (and t) ---
        self.kv_proj = nn.Linear(1536, D)
        self.kv_norm = nn.LayerNorm(D)

        # --- Optional: project full text sequence to D for richer KV ---
        if self.use_text_seq:
            self.txt_proj = nn.Linear(4096, D)

        # --- 1) Self-attention over latent tokens (one lightweight block) ---
        self.self_attn = nn.MultiheadAttention(D, num_heads=H, batch_first=True, dropout=dropout)
        self.self_ffn  = nn.Sequential(
            nn.Linear(D, 4*D), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*D, D), nn.LayerNorm(D)
        )

        # --- 2) Cross-attention: latent tokens (Q) attend to KV = [y_tok, t_tok, (text_seq?)] ---
        self.cross_attn = nn.MultiheadAttention(D, num_heads=H, batch_first=True, dropout=dropout)
        self.cross_ffn  = nn.Sequential(
            nn.Linear(D, 4*D), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*D, D), nn.LayerNorm(D)
        )

        # --- Head: small MLP -> unit vector * positive magnitude (stable modulation power control) ---
        self.out_proj = nn.Sequential(
            nn.Linear(D, 4*D), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*D, D), nn.LayerNorm(D)
        )
        self.register_parameter("q_weight", nn.Parameter(torch.ones(1) * 0.1))

        self.pool_q = nn.Parameter(torch.randn(1, 1, D) * 0.02)  # learnable query

    def _attn_pool(self, x):
        # x: [B, L, D]
        q = self.pool_q.expand(x.size(0), -1, -1)             # [B,1,D]
        attn = torch.matmul(q, x.transpose(1, 2)) / (x.size(-1) ** 0.5)  # [B,1,L]
        w = attn.softmax(dim=-1)
        return torch.matmul(w, x).squeeze(1)                  # [B,D]


    @staticmethod
    def _build_2d_sincos_pos_embed(dim, hw):
        def _pos1d(n, d):
            pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)      # [n,1]
            i   = torch.arange(d//2, dtype=torch.float32).unsqueeze(0)   # [1,d/2]
            omega = 1.0 / (10000 ** (i / (d//2)))
            angle = pos @ omega
            return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)  # [n,d]
        px = _pos1d(hw, dim//2)
        py = _pos1d(hw, dim - dim//2)
        # tile (x,y) across grid; produces [L, dim], L=hw*hw
        pe_rows = []
        for y in range(hw):
            row = torch.cat([px, py[y].repeat(hw, 1)], dim=1)
            pe_rows.append(row)
        pe = torch.stack(pe_rows, dim=0).reshape(hw*hw, dim)
        return pe.unsqueeze(0)  # [1, L, dim]

    def forward(self, t, latent, y, c_crossattn=None):
        B, C, H, W = latent.shape
        D = self.latent_proj.out_channels
        dtype = latent.dtype

        # --- KV: pooled prompt y (+ optional timestep) ---
        y_emb = self.y_embedder(y.to(dtype))                 # [B,1536]
        y_tok = self.kv_proj(y_emb).unsqueeze(1)             # [B,1,D]
        if self.time_bool:
            t_emb = self.t_embedder(t, dtype=dtype)          # [B,1536]
            t_tok = self.kv_proj(t_emb).unsqueeze(1)         # [B,1,D]
            kv = torch.cat([y_tok, t_tok], dim=1)            # [B,2,D]
        else:
            kv = y_tok                                       # [B,1,D]
        if self.use_text_seq and (c_crossattn is not None):
            kv_txt = self.txt_proj(c_crossattn.to(dtype))    # [B,L,D]
            kv = torch.cat([kv, kv_txt], dim=1)              # [B,2+L,D]
        kv = self.kv_norm(kv)

        # --- Latent -> tokens + pos enc ---
        x = self.latent_proj(latent)                         # [B,D,H,W]
        x = self.down(x)                                     # [B,D,K,K]
        x = x.flatten(2).transpose(1, 2)                     # [B,L,D], L=K*K
        x = x + self.pos_embed.to(x.device, x.dtype)         # add 2D sin-cos
        x = self.latent_norm(x)

        # --- (1) Self-attention over latent tokens ---
        sa_out, _ = self.self_attn(x, x, x)
        x = x + sa_out
        x = x + self.self_ffn(x)

        # --- (2) Cross-attention: latent queries attend to KV (y,t,(text_seq?)) ---
        ca_out, _ = self.cross_attn(x, kv, kv)
        x = x + ca_out
        x = x + self.cross_ffn(x)

        # --- Pool tokens -> single vector ---
        x = self._attn_pool(x)                               # [B,D]

        # --- Head: direction * magnitude ---
        q_raw = self.out_proj(x)                             # [B,D]
        q_dir = F.normalize(q_raw, p=2, dim=-1, eps=1e-6)    # unit direction
        scale = F.softplus(self.q_weight).to(q_dir.dtype)    # shared ≥ 0 scalar
        q_t = scale * q_dir                                  # [B,D]
        return q_t
    
class QRMModulatorLatentV2(nn.Module):
    """
    Inputs:
      x            : [B, C, H, W] latent image
      scale_shift  : [B, D_all] concatenated baseline mod vectors (all blocks + final)
      block_spans  : list of (start, end) index pairs slicing scale_shift into per-block chunks
                     in the SAME order you assigned _qrm_offset in BaseModel.
    Output:
      qrm_delta    : [B, D_all] additive correction aligned with scale_shift layout
    """
    def __init__(
        self,
        block_spans,
        d_model=384, # 384 6 2 2
        n_heads=6,
        n_layers=1,
        latent_grid=1,
        mlp_ratio=4.0,
        dropout=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.block_spans = list(block_spans)
        self.M = len(self.block_spans)
        self.d_model = d_model

        # -------- latent encoder -> P tokens --------
        # 1x1 conv to d_model, then AdaptiveAvgPool to SxS and flatten
        self._latent_in_channels = 16
        self.latent_proj = nn.Conv2d(16, self.d_model, kernel_size=1, bias=True)

        self.latent_grid = latent_grid  # S

        # -------- per-block input projections (native m_i -> d_model) --------
        in_projs = []
        out_projs = []
        for (s, e) in self.block_spans:
            m_i = e - s
            in_projs.append(nn.Linear(m_i, d_model))
            proj_out = nn.Linear(d_model, m_i)
            nn.init.trunc_normal_(proj_out.weight,std=1e-3)
            nn.init.zeros_(proj_out.bias)
            out_projs.append(proj_out)
        self.in_projs = nn.ModuleList(in_projs)
        self.out_projs = nn.ModuleList(out_projs)

        # token pos-emb for modulation tokens
        self.mod_pos = nn.Parameter(torch.zeros(self.M, d_model))
        nn.init.trunc_normal_(self.mod_pos, std=0.02)

        # per-token gates (start near 0 influence)
        init_gate_p = 0.25  # (0,1)
        logit_p = math.log(init_gate_p) - math.log1p(1.0 - init_gate_p)
        self.gate_bias = nn.Parameter(torch.full((self.M,), logit_p))

        # -------- tiny cross-attentive transformer over modulation tokens --------
        # Pre-LN, residual MHA (queries=mod tokens, KV=latent tokens), then FFN
        layers = []
        for _ in range(n_layers):
            layers.append(_CrossAttnBlockV2(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
                                          dropout=dropout, ln_eps=ln_eps))
        self.layers = nn.ModuleList(layers)

        self.pre_out_ln = nn.LayerNorm(d_model, eps=ln_eps)

    def _latent_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.latent_proj(x)
        h = F.adaptive_avg_pool2d(h, output_size=self.latent_grid)
        return h.flatten(2).transpose(1, 2)

    def _mod_tokens(self, scale_shift: torch.Tensor) -> torch.Tensor:
        """
        Split scale_shift into per-block chunks and project to tokens: [B, M, d]
        """
        zs = []
        for (proj, (s, e)) in zip(self.in_projs, self.block_spans):
            u = scale_shift[:, s:e]         # [B, m_i]
            u = F.layer_norm(u, u.shape[-1:])
            z = proj(u)                     # [B, d]
            zs.append(z)
        Z = torch.stack(zs, dim=1)          # [B, M, d]
        return Z

    def forward(self, x: torch.Tensor, scale_shift: torch.Tensor) -> torch.Tensor:
        # channels_last for conv efficiency
        x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda"):
            X = self._latent_tokens(x)            # [B,P,d]
            Z = self._mod_tokens(scale_shift)     # [B,M,d]
            Z = Z + self.mod_pos.unsqueeze(0)
            for blk in self.layers:
                Z = blk(Z, X)
            Z = self.pre_out_ln(Z)
            B         = scale_shift.shape[0]
            sum_m     = self.block_spans[-1][1]
            qrm_delta = torch.empty(B, sum_m, device=Z.device, dtype=Z.dtype)

            gates = torch.sigmoid(self.gate_bias).view(1, self.M, 1)
            for i, proj_out in enumerate(self.out_projs):
                s, e = self.block_spans[i]
                qrm_delta[:, s:e] = proj_out(Z[:, i, :]) * gates[:, i, :]

            return qrm_delta

#part of QRMModulatorLatentV2
class _CrossAttnBlockV2(nn.Module):
    """
    Pre-LN cross-attention block:
      Z <- Z + MHA( LN(Z) as queries, LN(X) as KV )
      Z <- Z + MLP( LN(Z) )
    """
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0, ln_eps=1e-6):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.kv_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.ff_ln = nn.LayerNorm(d_model, eps=ln_eps)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        # zero-init the last FF layer to keep block near identity at start
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        q = self.q_ln(Z)
        kv = self.kv_ln(X)
        a, _ = self.mha(q, kv, kv, need_weights=False)   # [B, M, d]
        Z = Z + self.attn_drop(a)
        # FFN
        Z = Z + self.ff(self.ff_ln(Z))
        return Z
    

class QRMModulatorLatentV3(nn.Module):
    """
    Inputs
      x               : [B, 16, H, W]  latent image
      scale_shift     : [B, D_all]     concatenated baseline adaLN vectors (same layout as V2)
      prompt_embedding: [B, P]         pooled prompt embedding (e.g., SD3 pooled y or CLIP pooled; set P accordingly)
      timestep_embedding: [B, P]       Timestep embedding
    Output
      qrm_delta       : [B, D_all]     additive correction aligned with scale_shift layout
    """
    def __init__(
        self,
        block_spans,
        prompt_dim=1536,
        d_model=384,
        n_heads=6,
        n_layers=1,
        latent_grid=1,
        mlp_ratio=4.0,
        dropout=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.block_spans = list(block_spans)
        self.M = len(self.block_spans)
        self.d_model = d_model
        self.latent_grid = latent_grid
        self.y_embedder = VectorEmbedder(2048, 1536)
        self.t_embedder = TimestepEmbedder(1536)

        self.latent_proj = nn.Conv2d(16, d_model, kernel_size=1, bias=True)

        in_projs, out_projs = [], []
        for (s, e) in self.block_spans:
            m_i = e - s
            in_projs.append(nn.Linear(m_i, d_model))
            proj_out = nn.Linear(d_model, m_i)
            nn.init.trunc_normal_(proj_out.weight, std=1e-3)
            nn.init.zeros_(proj_out.bias)
            out_projs.append(proj_out)
        self.in_projs = nn.ModuleList(in_projs)
        self.out_projs = nn.ModuleList(out_projs)

        self.mod_pos = nn.Parameter(torch.zeros(self.M, d_model))
        nn.init.trunc_normal_(self.mod_pos, std=0.02)

        self.prompt_ln   = nn.LayerNorm(prompt_dim, eps=ln_eps)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)
        self.t_proj = nn.Linear(1536, self.d_model)

        layers = []
        for _ in range(n_layers):
            layers.append(_CrossAttnBlockV3(d_model=d_model, n_heads=n_heads,
                                          mlp_ratio=mlp_ratio, dropout=dropout, ln_eps=ln_eps))
        self.layers = nn.ModuleList(layers)
        self.pre_out_ln = nn.LayerNorm(d_model, eps=ln_eps)

    # ---------- helpers ----------
    def _latent_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(x)                                 # [B,d,H,W]
        h = F.adaptive_avg_pool2d(h, output_size=self.latent_grid)
        return h.flatten(2).transpose(1, 2)                     # [B,P,d]

    def _mod_tokens(self, scale_shift: torch.Tensor) -> torch.Tensor:
        zs = []
        for (proj, (s, e)) in zip(self.in_projs, self.block_spans):
            u = scale_shift[:, s:e]                              # [B, m_i]
            u = F.layer_norm(u, u.shape[-1:])
            z = proj(u)                                          # [B, d]
            zs.append(z)
        return torch.stack(zs, dim=1)                            # [B, M, d]

    # ---------- forward ----------
    def forward(
        self,
        x: torch.Tensor,
        scale_shift: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda"):
            # 1) tokens
            X = self._latent_tokens(x)                           # [B,P,d]
            Z = self._mod_tokens(scale_shift)                    # [B,M,d]
            Z = Z + self.mod_pos.unsqueeze(0)                    # add PE

            # 2) build KV = [latent tokens || prompt token]
            y_emb = self.y_embedder(y)
            y_tok = self.prompt_proj(self.prompt_ln(y_emb)).unsqueeze(1)  # [B,1,d]
            t_emb = self.t_embedder(timestep,dtype=Z.dtype)
            t_tok = self.t_proj(t_emb).unsqueeze(1) 
            KV = torch.cat([X, y_tok,t_tok], dim=1)                    # [B,P+1,d]

            for blk in self.layers:
                Z = blk(Z, KV)
            Z = self.pre_out_ln(Z)                               # [B,M,d]

            B         = scale_shift.shape[0]
            sum_m     = self.block_spans[-1][1]
            qrm_delta = torch.empty(B, sum_m, device=Z.device, dtype=Z.dtype)



            for i, proj_out in enumerate(self.out_projs):
                s, e = self.block_spans[i]
                qrm_delta[:, s:e] = proj_out(Z[:, i, :])

            return qrm_delta


class _CrossAttnBlockV3(nn.Module):
    """
    Pre-LN cross-attention block:
      Z <- Z + MHA( LN(Z) as queries, LN(KV) as K=V )
      Z <- Z + MLP( LN(Z) )
    """
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0, ln_eps=1e-6):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.kv_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_ln = nn.LayerNorm(d_model, eps=ln_eps)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, Z: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
        q  = self.q_ln(Z)
        kv = self.kv_ln(KV)
        a, _ = self.mha(q, kv, kv, need_weights=False)   # [B,M,d]
        Z = Z + self.attn_drop(a)
        Z = Z + self.ff(self.ff_ln(Z))
        return Z
    
class QRMModulatorLatentV4(nn.Module):
    """
    Inputs
      x               : [B, 16, H, W]  latent image
      scale_shift     : [B, D_all]     concatenated baseline adaLN vectors (same layout as V2)
      prompt_embedding: [B, P]         pooled prompt embedding (e.g., SD3 pooled y or CLIP pooled; set P accordingly)
      timestep_embedding: [B, P]       Timestep embedding
    Output
      qrm_delta       : [B, D_all]     additive correction aligned with scale_shift layout
    """
    def __init__(
        self,
        block_spans,
        prompt_dim=1536,
        d_model=384,
        n_heads=6,
        n_layers=2,
        latent_grid=2,
        mlp_ratio=4.0,
        dropout=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.block_spans = list(block_spans)
        self.M = len(self.block_spans)
        self.d_model = d_model
        self.latent_grid = latent_grid
        self.y_embedder = VectorEmbedder(2048, 1536)
        self.t_embedder = TimestepEmbedder(1536)

        self.latent_proj = nn.Conv2d(16, d_model, kernel_size=1, bias=True)

        in_projs, out_projs = [], []
        for (s, e) in self.block_spans:
            m_i = e - s
            in_projs.append(nn.Linear(m_i, d_model))
            proj_out = nn.Linear(d_model, m_i)
            nn.init.trunc_normal_(proj_out.weight, std=1e-3)
            nn.init.zeros_(proj_out.bias)
            out_projs.append(proj_out)
        self.in_projs = nn.ModuleList(in_projs)
        self.out_projs = nn.ModuleList(out_projs)

        self.mod_pos = nn.Parameter(torch.zeros(self.M, d_model))
        nn.init.trunc_normal_(self.mod_pos, std=0.02)

        self.prompt_ln   = nn.LayerNorm(prompt_dim, eps=ln_eps)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)
        self.t_proj = nn.Linear(1536, self.d_model)

        layers = []
        for _ in range(n_layers):
            layers.append(_DecoderBlockV3(d_model=d_model, n_heads=n_heads,
                                          mlp_ratio=mlp_ratio, dropout=dropout, ln_eps=ln_eps))
        self.layers = nn.ModuleList(layers)
        self.pre_out_ln = nn.LayerNorm(d_model, eps=ln_eps)

    # ---------- helpers ----------
    def _latent_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(x)                                 # [B,d,H,W]
        h = F.adaptive_avg_pool2d(h, output_size=self.latent_grid)
        return h.flatten(2).transpose(1, 2)                     # [B,P,d]

    def _mod_tokens(self, scale_shift: torch.Tensor) -> torch.Tensor:
        zs = []
        for (proj, (s, e)) in zip(self.in_projs, self.block_spans):
            u = scale_shift[:, s:e]                              # [B, m_i]
            u = F.layer_norm(u, u.shape[-1:])
            z = proj(u)                                          # [B, d]
            zs.append(z)
        return torch.stack(zs, dim=1)                            # [B, M, d]

    # ---------- forward ----------
    def forward(
        self,
        x: torch.Tensor,
        scale_shift: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda"):
            # 1) tokens
            X = self._latent_tokens(x)                           # [B,P,d]
            Z = self._mod_tokens(scale_shift)                    # [B,M,d]
            Z = Z + self.mod_pos.unsqueeze(0)                    # add PE

            # 2) build KV = [latent tokens || prompt token]
            y_emb = self.y_embedder(y)
            y_tok = self.prompt_proj(self.prompt_ln(y_emb)).unsqueeze(1)  # [B,1,d]
            t_emb = self.t_embedder(timestep,dtype=Z.dtype)
            t_tok = self.t_proj(t_emb).unsqueeze(1) 
            KV = torch.cat([X, y_tok,t_tok], dim=1)                    # [B,P+1,d]

            for blk in self.layers:
                Z = blk(Z, KV)
            Z = self.pre_out_ln(Z)                               # [B,M,d]

            B         = scale_shift.shape[0]
            sum_m     = self.block_spans[-1][1]
            qrm_delta = torch.empty(B, sum_m, device=Z.device, dtype=Z.dtype)



            for i, proj_out in enumerate(self.out_projs):
                s, e = self.block_spans[i]
                qrm_delta[:, s:e] = proj_out(Z[:, i, :])

            return qrm_delta
        
class QRMModulatorLatentV5(nn.Module):
    """
    Inputs
      x               : [B, 16, H, W]  latent image
      scale_shift     : [B, D_all]     concatenated baseline adaLN vectors (same layout as V2)
      prompt_embedding: [B, P]         pooled prompt embedding (e.g., SD3 pooled y or CLIP pooled; set P accordingly)
      timestep_embedding: [B, P]       Timestep embedding
    Output
      qrm_delta       : [B, D_all]     additive correction aligned with scale_shift layout
    """
    def __init__(
        self,
        block_spans,
        prompt_dim=1536,
        d_model=512,
        n_heads=8,
        n_layers=4,
        latent_grid=2,
        mlp_ratio=3.5,
        dropout=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.block_spans = list(block_spans)
        self.M = len(self.block_spans)
        self.d_model = d_model
        self.latent_grid = latent_grid
        self.y_embedder = VectorEmbedder(2048, 1536)
        self.t_embedder = TimestepEmbedder(1536)

        self.latent_proj = nn.Conv2d(16, d_model, kernel_size=1, bias=True)

        in_projs, out_projs = [], []
        for (s, e) in self.block_spans:
            m_i = e - s
            in_projs.append(nn.Linear(m_i, d_model))
            proj_out = nn.Linear(d_model, m_i)
            nn.init.trunc_normal_(proj_out.weight, std=1e-3)
            nn.init.zeros_(proj_out.bias)
            out_projs.append(proj_out)
        self.in_projs = nn.ModuleList(in_projs)
        self.out_projs = nn.ModuleList(out_projs)

        self.mod_pos = nn.Parameter(torch.zeros(self.M, d_model))
        nn.init.trunc_normal_(self.mod_pos, std=0.02)

        self.prompt_ln   = nn.LayerNorm(prompt_dim, eps=ln_eps)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)
        self.t_proj = nn.Linear(1536, self.d_model)

        layers = []
        for _ in range(n_layers):
            layers.append(_DecoderBlockV3(d_model=d_model, n_heads=n_heads,
                                          mlp_ratio=mlp_ratio, dropout=dropout, ln_eps=ln_eps))
        self.layers = nn.ModuleList(layers)
        self.pre_out_ln = nn.LayerNorm(d_model, eps=ln_eps)

    # ---------- helpers ----------
    def _latent_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(x)                                 # [B,d,H,W]
        h = F.adaptive_avg_pool2d(h, output_size=self.latent_grid)
        return h.flatten(2).transpose(1, 2)                     # [B,P,d]

    def _mod_tokens(self, scale_shift: torch.Tensor) -> torch.Tensor:
        zs = []
        for (proj, (s, e)) in zip(self.in_projs, self.block_spans):
            u = scale_shift[:, s:e]                              # [B, m_i]
            u = F.layer_norm(u, u.shape[-1:])
            z = proj(u)                                          # [B, d]
            zs.append(z)
        return torch.stack(zs, dim=1)                            # [B, M, d]

    # ---------- forward ----------
    def forward(
        self,
        x: torch.Tensor,
        scale_shift: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda"):
            # 1) tokens
            X = self._latent_tokens(x)                           # [B,P,d]
            Z = self._mod_tokens(scale_shift)                    # [B,M,d]
            Z = Z + self.mod_pos.unsqueeze(0)                    # add PE

            # 2) build KV = [latent tokens || prompt token]
            y_emb = self.y_embedder(y)
            y_tok = self.prompt_proj(self.prompt_ln(y_emb)).unsqueeze(1)  # [B,1,d]
            t_emb = self.t_embedder(timestep,dtype=Z.dtype)
            t_tok = self.t_proj(t_emb).unsqueeze(1) 
            KV = torch.cat([X, y_tok,t_tok], dim=1)                    # [B,P+1,d]

            for blk in self.layers:
                Z = blk(Z, KV)
            Z = self.pre_out_ln(Z)                               # [B,M,d]

            B         = scale_shift.shape[0]
            sum_m     = self.block_spans[-1][1]
            qrm_delta = torch.empty(B, sum_m, device=Z.device, dtype=Z.dtype)



            for i, proj_out in enumerate(self.out_projs):
                s, e = self.block_spans[i]
                qrm_delta[:, s:e] = proj_out(Z[:, i, :])

            return qrm_delta
        
class QRMModulatorLatentV6(nn.Module):
    """
    Inputs
      x               : [B, 16, H, W]  latent image
      scale_shift     : [B, D_all]     concatenated baseline adaLN vectors (same layout as V2)
      prompt_embedding: [B, P]         pooled prompt embedding (e.g., SD3 pooled y or CLIP pooled; set P accordingly)
      timestep_embedding: [B, P]       Timestep embedding
    Output
      qrm_delta       : [B, D_all]     additive correction aligned with scale_shift layout
    """
    def __init__(
        self,
        block_spans,
        prompt_dim=1536,
        d_model=640,
        n_heads=10,
        n_layers=6,
        latent_grid=4,
        mlp_ratio=3.0,
        dropout=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.block_spans = list(block_spans)
        self.M = len(self.block_spans)
        self.d_model = d_model
        self.latent_grid = latent_grid
        self.y_embedder = VectorEmbedder(2048, 1536)
        self.t_embedder = TimestepEmbedder(1536)

        self.latent_proj = nn.Conv2d(16, d_model, kernel_size=1, bias=True)

        in_projs, out_projs = [], []
        for (s, e) in self.block_spans:
            m_i = e - s
            in_projs.append(nn.Linear(m_i, d_model))
            proj_out = nn.Linear(d_model, m_i)
            nn.init.trunc_normal_(proj_out.weight, std=1e-3)
            nn.init.zeros_(proj_out.bias)
            out_projs.append(proj_out)
        self.in_projs = nn.ModuleList(in_projs)
        self.out_projs = nn.ModuleList(out_projs)

        self.mod_pos = nn.Parameter(torch.zeros(self.M, d_model))
        nn.init.trunc_normal_(self.mod_pos, std=0.02)

        self.prompt_ln   = nn.LayerNorm(prompt_dim, eps=ln_eps)
        self.prompt_proj = nn.Linear(prompt_dim, d_model)
        self.t_proj = nn.Linear(1536, self.d_model)

        layers = []
        for _ in range(n_layers):
            layers.append(_DecoderBlockV3(d_model=d_model, n_heads=n_heads,
                                          mlp_ratio=mlp_ratio, dropout=dropout, ln_eps=ln_eps))
        self.layers = nn.ModuleList(layers)
        self.pre_out_ln = nn.LayerNorm(d_model, eps=ln_eps)

    # ---------- helpers ----------
    def _latent_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(x)                                 # [B,d,H,W]
        h = F.adaptive_avg_pool2d(h, output_size=self.latent_grid)
        return h.flatten(2).transpose(1, 2)                     # [B,P,d]

    def _mod_tokens(self, scale_shift: torch.Tensor) -> torch.Tensor:
        zs = []
        for (proj, (s, e)) in zip(self.in_projs, self.block_spans):
            u = scale_shift[:, s:e]                              # [B, m_i]
            u = F.layer_norm(u, u.shape[-1:])
            z = proj(u)                                          # [B, d]
            zs.append(z)
        return torch.stack(zs, dim=1)                            # [B, M, d]

    # ---------- forward ----------
    def forward(
        self,
        x: torch.Tensor,
        scale_shift: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda"):
            # 1) tokens
            X = self._latent_tokens(x)                           # [B,P,d]
            Z = self._mod_tokens(scale_shift)                    # [B,M,d]
            Z = Z + self.mod_pos.unsqueeze(0)                    # add PE

            # 2) build KV = [latent tokens || prompt token]
            y_emb = self.y_embedder(y)
            y_tok = self.prompt_proj(self.prompt_ln(y_emb)).unsqueeze(1)  # [B,1,d]
            t_emb = self.t_embedder(timestep,dtype=Z.dtype)
            t_tok = self.t_proj(t_emb).unsqueeze(1) 
            KV = torch.cat([X, y_tok,t_tok], dim=1)                    # [B,P+1,d]

            for blk in self.layers:
                Z = blk(Z, KV)
            Z = self.pre_out_ln(Z)                               # [B,M,d]

            B         = scale_shift.shape[0]
            sum_m     = self.block_spans[-1][1]
            qrm_delta = torch.empty(B, sum_m, device=Z.device, dtype=Z.dtype)



            for i, proj_out in enumerate(self.out_projs):
                s, e = self.block_spans[i]
                qrm_delta[:, s:e] = proj_out(Z[:, i, :])

            return qrm_delta


class _DecoderBlockV3(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0, ln_eps=1e-6):
        super().__init__()
        # self-attn
        self.self_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        # cross-attn
        self.cross_q_ln  = nn.LayerNorm(d_model, eps=ln_eps)
        self.cross_kv_ln = nn.LayerNorm(d_model, eps=ln_eps)
        self.cross_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        # ff
        self.ff_ln = nn.LayerNorm(d_model, eps=ln_eps)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.ff[-1].weight); nn.init.zeros_(self.ff[-1].bias)

    def forward(self, Z, KV):
        # self-attn over Z (non-causal)
        z_norm = self.self_ln(Z)
        s, _ = self.self_attn(z_norm, z_norm, z_norm, need_weights=False)
        Z = Z + s
        # cross-attn to KV
        q = self.cross_q_ln(Z); kv = self.cross_kv_ln(KV)
        a, _ = self.cross_attn(q, kv, kv, need_weights=False)
        Z = Z + a
        # feedforward
        Z = Z + self.ff(self.ff_ln(Z))
        return Z


    
QRMRegistry = {
    "QRMModulatorLatent":QRMModulatorLatent,
    "QRMModulatorLatentV2":QRMModulatorLatentV2,
    "QRMModulatorLatentV3": QRMModulatorLatentV3,
    "QRMModulatorLatentV4": QRMModulatorLatentV4,
    "QRMModulatorLatentV5": QRMModulatorLatentV5,
    "QRMModulatorLatentV6": QRMModulatorLatentV6,
}