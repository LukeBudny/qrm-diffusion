import torch
import torch.nn as nn
from other_impls import Mlp
import torch
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

class QRMMLP(nn.Module):
    def __init__(
        self,
        # t_embedder: nn.Module,
        # y_embedder: nn.Module,
        hidden_dim: int = 1024,
        out_dim: int = 1536,
        vision_dim: int = 512,
        time_bool = True
    ):
        super().__init__()
        self.time_bool = time_bool
        if self.time_bool:
            self.t_embedder = TimestepEmbedder(1536, dtype=torch.float32)

        self.y_embedder = VectorEmbedder(2048, 1536, dtype=torch.float32)
        t_embed_dim = 1536 if time_bool else 0
        y_embed_dim = 1536

        # Final fusion MLP: receives vision + t_emb + y_emb
        self.mlp = Mlp(
            in_features=vision_dim + t_embed_dim + y_embed_dim,
            hidden_features=hidden_dim,
            out_features=out_dim,
            act_layer=nn.GELU,
            dtype=torch.float32
        )

        self.scale = nn.Parameter(torch.tensor(2.5))

    def forward(self, t: torch.Tensor, vision_feature: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dtype = vision_feature.dtype
        # Embed time using shared embedder

        # Embed pooled y (e.g. from T5/CLIP)
        y_emb = self.y_embedder(y.to(dtype))                  # [B, 2048]

        # Concat everything
        if self.time_bool:
            t_emb = self.t_embedder(t, dtype=dtype)               # [B, 2048]
            x = torch.cat([vision_feature.to(dtype), t_emb, y_emb], dim=-1)  # [B, 512+2048+2048]
        else:
            x = torch.cat([vision_feature.to(dtype), y_emb], dim=-1)  # [B, 512+2048+2048]
        q_raw = self.mlp(x)
        q_normed = F.normalize(q_raw, p=2, dim=-1, eps=1e-6)
        q_t = q_normed * self.scale
        # self.q_t_last = q_t.detach()
        return q_t                                           # float32
    
class QRMTransformer(nn.Module):
    def __init__(self, out_dim=1536, time_bool=True,dropout=0.1,vision_dim: int = 512):
        super().__init__()
        self.time_bool = time_bool
        self.t_embedder = TimestepEmbedder(1536) if time_bool else None
        self.y_embedder = VectorEmbedder(2048, 1536)

        self.q_proj = nn.Linear(vision_dim, out_dim)
        self.kv_proj = nn.Linear(1536 + (1536 if time_bool else 0), out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True,dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),               
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.register_parameter("q_weight", nn.Parameter(torch.ones(1)))

    def forward(self, t: torch.Tensor, vision_feature: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dtype = vision_feature.dtype
        y_emb = self.y_embedder(y.to(dtype))
        t_emb = self.t_embedder(t, dtype=dtype) if self.time_bool else None
        kv = torch.cat([y_emb, t_emb], dim=-1) if t_emb is not None else y_emb
        kv = self.kv_proj(kv).unsqueeze(1)  # [B, 1, D]

        q = self.q_proj(vision_feature.to(dtype)).unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(q, kv, kv)                      # [B, 1, D]
        q_t = self.out_proj(attn_out + q).squeeze(1)

        q_t = self.q_weight.to(q_t.dtype) * q_t
        return q_t

QRMRegistry = {
    "mlp": QRMMLP,
    "transformer": QRMTransformer,
}