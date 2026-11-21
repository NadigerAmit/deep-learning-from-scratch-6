# Concept: key/value caching reuses past attention projections during autoregressive decoding.
# Goal: contrast repeated full-prefix inference with one-token cached inference.
# The script measures both approaches on the same randomly initialized GPT shape.
# KV-cache state is maintained separately inside each attention block.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0
        half = key_dim // 2

        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ( (2.0 * half_ids) / key_dim ))  # (half,)

        positions = torch.arange(max_context_len)  # (max_context_len,)
        angles = positions[:, None] * inv_freq[None, :]  # (max_context_len, half)

        cos = torch.cos(angles)  # (max_context_len, half)
        sin = torch.sin(angles)  # (max_context_len, half)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x, offset=0):
        batch_size, num_head, context_len, key_dim = x.shape

        input_dtype = x.dtype
        x = x.float()

        # Select RoPE rows at the current absolute offset, not from position zero.
        max_context_len = self.cos_cache.size(0)
        if offset + context_len > max_context_len:
            offset = max_context_len - context_len

        cos = self.cos_cache[offset:offset + context_len]
        sin = self.sin_cache[offset:offset + context_len]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        out = out.reshape(batch_size, num_head, context_len, key_dim)

        return out.to(input_dtype)


# KV means key and value: past projections do not change during decoding. Reuse avoids recomputation;
# memory grows with B*H*past_tokens*D for each cached tensor.
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, rope=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.rope = rope

        # Per-layer inference state: cached K/V tensors have shape (B,H,past_tokens,D).
        self.k_cache = None  # Cached keys for earlier tokens.
        self.v_cache = None  # Reuse past values instead of recomputing the prefix.
        self.cache_offset = 0  # Absolute position of the next token.

    def forward(self, x, use_cache=False):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # Pass the absolute cache offset to RoPE so new tokens receive true positions.
        if self.rope is not None:
            if use_cache:
                Q = self.rope(Q, self.cache_offset)
                K = self.rope(K, self.cache_offset)
            else:
                Q = self.rope(Q)
                K = self.rope(K)

        # Cache tensors are ordinary device tensors, not trainable parameters.
        # Handle cached keys/values for autoregressive decoding.
        if use_cache:
            # First call is prompt prefill; later calls append newly projected token(s).
            is_first_call = (self.k_cache is None)

            if is_first_call:
                # First call: initialize the cache with prompt keys and values.
                self.k_cache = K
                self.v_cache = V
            else:
                # Later calls: append newly projected keys and values along the sequence axis.
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)

            # Advance the absolute token position for the next RoPE row.
            self.cache_offset += C

            # Use all keys and values accumulated so far.
            K = self.k_cache
            V = self.v_cache

        # Compute scaled dot-product attention over current queries and available keys.
        # Cached shapes: Q=(B,H,new,D), K=(B,H,past+new,D), scores=(B,H,new,past+new).
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D ** 0.5)

        # Prefill needs a causal mask; a one-token decode query can see cached/current keys.
        if not use_cache or is_first_call:
            mask = torch.tril(torch.ones(C, C, device=scores.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

    def clear_cache(self):
        """Discard cached keys and values before an independent prompt."""
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)

        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)

    def forward(self, x):
        a = self.W(x)
        b = self.V(x)

        gated = F.silu(a) * b  # silu(a) * b
        out = self.O(gated)
        return out

class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_dim)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ffn(self.norm2(x))
        return x

    def clear_cache(self):
        """Clear this block’s attention cache."""
        self.attn.clear_cache()


class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, theta=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.theta = theta

        self.embed = nn.Embedding(vocab_size, embed_dim)

        head_dim = embed_dim // n_head
        rope = RoPE(theta, head_dim, max_context_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, rope)
            for _ in range(n_layer)
        ])

        self.norm = nn.RMSNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, use_cache=False):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        logits = self.unembed(x)
        return logits

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
            'theta': self.theta,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            theta=checkpoint['theta']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model

    def clear_cache(self):
        """Clear the key/value state in every transformer block."""
        for block in self.blocks:
            block.clear_cache()


# torch.no_grad prevents gradient storage because inference does not update weights.
def generate_without_cache(model, start_ids, max_new_tokens):
    model.eval()

    ids = start_ids  # Full generated prefix, initially the supplied token(s).
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Without caching, recompute the full prefix at every step.
            logits = model(ids, use_cache=False)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            # Append the selected next token to the growing prefix.
            ids = torch.cat([ids, next_id], dim=1)

    return ids

# Cached decoding feeds only the newest token after the initial call; IDs are concatenated for return.
def generate_with_cache(model, start_ids, max_new_tokens):
    model.eval()

    ids = start_ids  # Start the generated sequence with the caller-provided token(s).
    next_id = start_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # With caching, feed only the most recently generated token.
            logits = model(next_id, use_cache=True)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids

# time.time measures wall seconds; results vary by device, PyTorch build, and system load.
def measure_generation_time(model, start_ids, use_cache, num_tokens=200):
    if not use_cache:
        model.clear_cache()

    start_time = time.time()

    if use_cache:
        generate_with_cache(model, start_ids, num_tokens)
    else:
        generate_without_cache(model, start_ids, num_tokens)

    elapsed = time.time() - start_time
    return elapsed

# Timing demonstration (not a training run).
model = GPT(vocab_size=1000, max_context_len=256, embed_dim=384,
            n_head=6, n_layer=6, ff_dim=1024)


start_ids = torch.tensor([[42]])  # Fixed single token ID for this greedy-path timing comparison.

time_without = measure_generation_time(model, start_ids, use_cache=False)
time_with = measure_generation_time(model, start_ids, use_cache=True)
# Japanese output says “without KV cache” plus elapsed seconds.
print(f"KV-Cacheなし: {time_without:.2f}秒")
# Japanese output says “with KV cache” plus elapsed seconds.
print(f"KV-Cacheあり: {time_with:.2f}秒")
# Japanese output says “speed-up ratio,” formatted as a multiple.
print(f"高速化率: {time_without / time_with:.1f}倍")

# The disabled quoted example below contains Japanese console labels meaning “output comparison,” “with/without cache,” and “match.”
"""
print("\n=== 出力の一致確認 ===")
model.clear_cache()

# 同じ開始トークンで生成

# KV-Cacheなしで生成
output_without = generate_without_cache(model, start_ids, max_new_tokens=max_new_tokens)
print(f"KV-Cacheなし: {output_without[0, :11].tolist()}")

# KV-Cacheありで生成
model.clear_cache()
output_with = generate_with_cache(model, start_ids, max_new_tokens=max_new_tokens)

print(f"KV-Cacheあり: {output_with[0, :11].tolist()}")

print(output_with.shape, output_without.shape)
# 一致確認
if torch.equal(output_without[:, :max_new_tokens], output_with[:, :max_new_tokens]):
    print("✓ 出力が一致しました!")
else:
    print("✗ 出力が一致しません")
    print(f"差分の数: {(output_without[:, :max_new_tokens] != output_with[:, :max_new_tokens]).sum().item()}")
"""

# ===== CONCEPT NOTES =====
# CONCEPT: A KV cache stores already-computed keys and values so each decoding step projects only the new token and attends to cached history. The first call (prefill) processes the prompt with a causal mask; later one-token decode calls attend to all cached positions, so no additional causal mask is needed for that single query.
# GOAL: Show why caching reduces repeated work in autoregressive generation compared with recomputing the entire prefix at every step. This extends the GPT architecture with cache state and compares elapsed times.
# INPUT: A randomly initialized GPT uses vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6 (head_dim=64), n_layer=6, and ff_dim=1024. The fixed start_ids tensor is [[42]] with shape (1,1); each timing run generates 200 new tokens by argmax.
# OUTPUT: Prints elapsed seconds for the no-cache and cache runs, followed by their ratio formatted to one decimal place. The helper generation methods return IDs (here shape (1,201)), but the active script does not print them or save a model/image. A code block at the end containing an optional equality check is inside a triple-quoted string and is not executed.
# HOW INPUT BECOMES OUTPUT: 1) No-cache generation feeds all current IDs to the model on each of 200 iterations; its sequence grows from 1 to 201. 2) Cached generation feeds the initial token and then one new token each iteration. 3) In every attention layer, K and V are shaped (batch, heads, sequence, 64), appended along sequence dimension 2, while Q for the current input attends over the cache. 4) Both paths select argmax from the last-position logits and concatenate the next token. 5) time.time measures each complete run and the script prints time_without/time_with.
# NOTES: The measured speed ratio depends on CPU/GPU, PyTorch, and system load; it is not a fixed result. The demo builds a 6-layer model and generates 200 tokens, so CPU timing can be substantial. Cache must be cleared between independent prompts; this script clears it before the no-cache benchmark, while the cache benchmark starts with an empty cache in the freshly used model. The configured context is 256 and this 201-token result stays within it. No external files or GPU are required.
