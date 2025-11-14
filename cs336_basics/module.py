import torch
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(
            (out_features, in_features), 
            device=device, 
            dtype=dtype)
        )
        std = torch.sqrt(torch.tensor([2 / (in_features + out_features)]))
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=std,
            a=-3,
            b=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        result = einsum(x, self.W, "... in_dim, out_dim in_dim  -> ... out_dim")
        return result

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        return self.W[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

        x = x / rms
        result = x * self.W

        return result.to(in_dtype)

class FFN(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = Linear(d_model, d_ff)#torch.nn.Parameter(torch.empty((d_ff, d_model)))
        self.down_proj = Linear(d_model, d_ff)# torch.nn.Parameter(torch.empty((d_model, d_ff)))
        self.up_proj = Linear(d_ff, d_model)# torch.nn.Parameter(torch.empty((d_ff, d_model)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = gate * torch.sigmoid(gate) * up

        result = self.down_proj(hidden)
        return result

class Rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # hidden = torch.Tensor([pow(theta,(2 * k - 2) / d_k) for k in (1..d_k / 2)])
        indices = torch.arange(0, d_k//2,dtype=torch.float32)
        freqs = 1.0 / (theta ** (2 * indices / d_k))

        positions = torch.arange(0, max_seq_len)

        angles = einsum(positions, freqs, "seq_len, d_k_dived2 -> seq_len d_k_dived2")
        cos_cache = torch.cos(angles)  # [max_seq_len, dim//2]
        sin_cache = torch.sin(angles)  # [max_seq_len, dim//2]

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor :
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]

        pair_x = rearrange(x, "... (d_k_dived2 tow) -> ... d_k_dived2 tow", tow = 2)
        
        x1 = pair_x[..., 0]
        x2 = pair_x[..., 1]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin

        result = torch.stack([rotated_x1,rotated_x2],dim=-1)
        result = rearrange(result, "... d_k_dived2 tow -> ... (d_k_dived2 tow)")

        return result
    
def softmax(x: torch.Tensor, dim:int) -> torch.Tensor :
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = torch.tensor([K.shape[-1]])
    pre_softmax_val = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(d_k)

    if mask is not None :
        pre_softmax_val.masked_fill_(~mask, -torch.inf)

    attention_weights = softmax(pre_softmax_val, -1)

    return einsum(attention_weights, V, "... queries k_v, ... k_v d_v -> ... queries d_v")

class Attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None,
                enable_rope = False, theta: float = 10000, max_seq_len:int = 100):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.o_proj = Linear(num_heads * self.d_v, d_model)
        self.enable_rope = enable_rope
        if enable_rope :
            self.rope = Rope(theta, self.d_k, max_seq_len)

    def forward(
            self, 
            x: Float[Tensor, "... seq_len d_model"],
            token_positions = None
        ) -> Float[Tensor, "... seq_len d_model"] :
        seq_len = x.shape[-2]

        q: Float[Tensor, "... seq_len (num_head d_k)"] = self.q_proj(x)
        k: Float[Tensor, "... seq_len (num_head d_k)"] = self.k_proj(x)
        v: Float[Tensor, "... seq_len (num_head d_v)"] = self.v_proj(x)

        q = rearrange(q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", d_k = self.d_k)
        k = rearrange(k, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", d_k = self.d_k)
        v = rearrange(v, "... seq_len (num_head d_v) -> ... num_head seq_len d_v", d_v = self.d_v)
        if self.enable_rope :
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        result: Float[Tensor, "... num_head seq_len d_v"] = scaled_dot_product_attention(q, k, v, mask)
        result = rearrange(result, "... num_head seq_len d_v -> ... seq_len (num_head d_v)", d_v = self.d_v)
        out_put: Float[Tensor, "... seq_len d_model"] = self.o_proj(result)

        return out_put


