import torch
from torch import Tensor
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from cs336_basics.utils import softmax

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
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(
            (vocab_size, embedding_dim),
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

class Block(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            weights: dict[str, Tensor],
            max_seq_len: int,
            theta: float,
            num_layers: int = -1,
            
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.weights = weights
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.__init_parameters()

    def __init_parameters(self) :
        if self.num_layers == -1:
            layer_prefix = ""
        else:
            layer_prefix = f"layers.{self.num_layers}."

        self.norm1 = RMSNorm(self.d_model)
        self.norm1.W.data = self.weights[layer_prefix + "ln1.weight"]
        
        self.attn = Attention(self.d_model, self.num_heads, enable_rope=True, theta=self.theta, max_seq_len=self.max_seq_len)
        self.attn.q_proj.W.data = self.weights[layer_prefix + "attn.q_proj.weight"]
        self.attn.k_proj.W.data = self.weights[layer_prefix + "attn.k_proj.weight"]
        self.attn.v_proj.W.data = self.weights[layer_prefix + "attn.v_proj.weight"]
        self.attn.o_proj.W.data = self.weights[layer_prefix + "attn.output_proj.weight"]    

        self.norm2 = RMSNorm(self.d_model)
        self.norm2.W.data = self.weights[layer_prefix + "ln2.weight"]

        self.ffn = FFN(self.d_model, self.d_ff)
        self.ffn.gate_proj.W.data = self.weights[layer_prefix + "ffn.w1.weight"]
        self.ffn.down_proj.W.data = self.weights[layer_prefix + "ffn.w2.weight"]
        self.ffn.up_proj.W.data = self.weights[layer_prefix + "ffn.w3.weight"]
    def forward(
            self,
            in_features: Float[Tensor, " batch sequence_length d_model"],
            token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " batch sequence_length d_model"] :
        
        norm_features = self.norm1(in_features)

        attn_output = self.attn(norm_features, token_positions)
        in_features = in_features + attn_output

        norm_features = self.norm2(in_features)

        out_feature = in_features + self.ffn(norm_features)

        return out_feature


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    embedder = Embedding(vocab_size, d_model)
    embedder.W.data = weights["token_embeddings.weight"]

    in_features:Float[Tensor, "batch_size seq_len d_model"] = embedder(in_indices)

    token_positions: Int[Tensor, " ... sequence_length"] = repeat(
        torch.arange(in_features.size(1), device=in_features.device),
        'seq -> batch seq',
        batch=in_features.size(0)
    )

    layers = []
    for i in range(num_layers):
        layer = Block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            weights=weights,
            theta=rope_theta,
            max_seq_len=context_length,
            num_layers=i,
        )
        layers.append(layer)
    
    for layer in layers:
        in_features = layer(in_features,token_positions)
    
       
    # 5. 最终层归一化
    ln_final = RMSNorm(d_model)
    ln_final.W.data = weights["ln_final.weight"]
    in_features = ln_final(in_features)
    
    # 6. 语言模型头
    lm_head = Linear(d_model, vocab_size)
    lm_head.W.data = weights["lm_head.weight"]
    logits = lm_head(in_features)  # [batch_size, sequence_length, vocab_size]
    
    # 7. Softmax得到概率分布
    # output_probabilities = softmax(logits)
    
    return logits


