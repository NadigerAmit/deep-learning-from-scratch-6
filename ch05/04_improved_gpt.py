# Concept: a GPT decoder combining RoPE attention, RMSNorm, residuals, and SwiGLU.
# Goal: assemble the chapter's transformer components into a causal language model.
# The bottom-of-file smoke example performs one forward pass on random token IDs.
# Model save/load helpers serialize architecture settings together with weights.

import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE rotates paired Q/K coordinates by position: angle[p,j]=p/theta**(2*j/D), encoding relative offsets in attention.
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

    def forward(self, x):
        batch_size, num_head, context_len, key_dim = x.shape

        input_dtype = x.dtype
        x = x.float()

        cos = self.cos_cache[:context_len]
        sin = self.sin_cache[:context_len]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        out = out.reshape(batch_size, num_head, context_len, key_dim)

        return out.to(input_dtype)


# Input x=(B,C,E); Linear projections make Q/K/V, then reshape to (B,H,C,D); scores have shape (B,H,C,C).
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

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # RoPE
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        # torch.matmul computes query-key dot products; dividing by sqrt(D) controls softmax scale.
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D ** 0.5)

        # torch.tril builds a causal mask on the scores device; future-token scores become -inf.
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax makes allowed weights sum to one; multiplying by V mixes context.
        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden = hidden.view(B, C, H * D)
        output = self.W_o(hidden)
        return output

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

# Pre-norm block: RMSNorm before attention/MLP, with residual additions as identity paths.
# RMSNorm does not center as LayerNorm does; SwiGLU provides gated feed-forward features.
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Decoder-only GPT maps integer IDs (B,C) to next-token logits (B,C,V). Embedding looks up E-wide vectors.
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

    def forward(self, ids):
        # Token IDs (B,C) -> embeddings (B,C,E) -> transformer blocks -> vocabulary scores (B,C,V).
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.unembed(x)  # (B, C, vocab_size)
        return logits

    def save(self, file_path):
        # torch.save stores state_dict tensors and architecture metadata for reconstruction.
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
        # torch.load maps checkpoint tensors; model.to(device) moves parameters/buffers to CPU, GPU, or MPS.
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


# Example: six heads divide the 384-wide embedding into D=64 features each.
vocab_size = 10000
max_context_len = 256
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = int(embed_dim * 8 / 3)
theta = 10000

# Initialize random GPT weights; this forward smoke example does not train.
model = GPT(vocab_size, max_context_len, embed_dim, n_head,
            n_layer, ff_dim, theta)
# torch.randint samples IDs in [0,V); output has vocabulary scores per position.
batch_size = 8
dummy_input = torch.randint(0, vocab_size, (batch_size, max_context_len))
logits = model(dummy_input)
print(logits.shape)  # torch.Size([8, 256, 10000])

# ===== CONCEPT NOTES =====
# CONCEPT: This decoder-only GPT combines token embeddings, RoPE on Q/K, causal scaled dot-product multi-head attention, pre-normalization with RMSNorm, residual connections, SwiGLU feed-forward layers, a final RMSNorm, and a vocabulary projection. Its attention uses a lower-triangular mask so position t cannot read later tokens.
# GOAL: Integrate the previous component lessons (RoPE, SwiGLU, and RMSNorm) into a complete next-token model. It demonstrates the forward pass and provides save/load methods for later training and generation examples.
# INPUT: Top-level example settings: vocab_size=10000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6, ff_dim=int(384*8/3)=1024, theta=10000. It creates integer token IDs with shape (8,256), each in [0,9999].
# OUTPUT: Prints torch.Size([8, 256, 10000]), the logits for every position and vocabulary item. No checkpoint is saved by the demo; GPT.save(file_path) can save state_dict plus architecture metadata when called.
# HOW INPUT BECOMES OUTPUT: 1) Embed IDs (8,256) as vectors (8,256,384). 2) In each of 6 blocks, normalize, project Q/K/V, split 384 features into 6 heads of 64, rotate Q/K with RoPE, apply causal attention, join heads, and add the attention residual. 3) Normalize again, apply SwiGLU through a 1024-wide hidden layer, and add the feed-forward residual. 4) Apply final RMSNorm and project 384 features to 10000 logits, yielding (8,256,10000).
# NOTES: This file imports torch.nn.RMSNorm, so its availability depends on the installed PyTorch version. The model is randomly initialized in this example. Loading requires a compatible checkpoint with the metadata keys and state-dict names expected here (for example blocks.N.ffn, not a differently named blocks.N.mlp). CPU works but a full 6-layer forward is compute- and memory-intensive; no training or GPU is necessary for syntax validation.
