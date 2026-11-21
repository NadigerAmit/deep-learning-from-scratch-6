# Concept: a GPT-style language model turns token IDs into contextual vectors and then vocabulary scores.
# Token embeddings represent token identity; position embeddings tell the model where each token occurs.
# Causal Transformer blocks mix only past/current context, and the final linear layer emits logits.
# PyTorch nn supplies embeddings/layers; the shared Block is imported from codebot.model.

# Add the repository root to the Python import path for the shared codebot.model.Block.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # Resolve and enter the directory above ch02.
sys.path.append('.')  # Add the current directory to module search locations.

import torch
import torch.nn as nn
from codebot.model import Block


class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate):
        super().__init__()
        # Keep architecture settings so save/load can reconstruct the same model later.
        self.vocab_size = vocab_size            # Number of possible token IDs and output classes.
        self.max_context_len = max_context_len  # Maximum number of position embeddings available.
        self.embed_dim = embed_dim              # Feature width E used throughout the model.
        self.n_head = n_head                    # Number of parallel attention heads per block.
        self.n_layer = n_layer                  # Number of stacked Transformer blocks.
        self.ff_dim = ff_dim                    # Hidden feature width inside each block's FFN.
        self.dropout_rate = dropout_rate        # Fraction of activations randomly dropped during training.

        # Embedding layers convert integer IDs into trainable feature vectors.
        self.embed = nn.Embedding(vocab_size, embed_dim)  # Token IDs (B,C) -> vectors (B,C,E).
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)  # Position IDs -> vectors (C,E).
        self.dropout = nn.Dropout(dropout_rate)  # Regularization active in train mode only.

        # A ModuleList registers each block's parameters so PyTorch can train/save them.
        self.blocks = nn.ModuleList([  # nn.ModuleList registers each child block as part of this model.
            Block(embed_dim, n_head, ff_dim, dropout_rate)
            for _ in range(n_layer)
        ])

        # Final normalization and unembedding map each contextual E-vector to vocab_size logits.
        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        # Tie input token embeddings to output classifier weights: both names share one Parameter.
        # This reduces independent parameters and makes input/output token representations related.
        self.embed.weight = self.unembed.weight

        # Initialize linear/embedding weights with a small normal distribution.
        self.apply(self._init_weights)  # PyTorch Module.apply visits this module and all child modules.

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # In-place normal initialization of a weight matrix.
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # In-place zero initialization of the optional bias vector.
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):  # Integer token IDs have shape (B,C).
        B, C = ids.shape  # B is number of sequences; C is their current context length.
        device = ids.device  # Keep position indices on the same CPU/GPU device as ids.

        # Add token identity and position information before contextual processing.
        pos = torch.arange(0, C, dtype=torch.long, device=device)  # PyTorch positions [0,...,C-1], shape (C,).
        emb = self.embed(ids)  # nn.Embedding lookup: (B,C) integer IDs -> (B,C,E).
        pos_emb = self.pos_embed(pos)  # Position lookup: (C,) -> (C,E); broadcasts across B.
        x = self.dropout(emb + pos_emb)  # Same-shape sum (B,C,E), then training-time dropout.

        # Each causal Transformer block updates every position using only permitted context.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # LayerNorm stabilizes final features; unembed is nn.Linear applied to each token.
        logits = self.unembed(x)  # (B,C,E) -> (B,C,vocab_size), unnormalized class scores.
        return logits

    def save(self, file_path):
        # Save both learned state and constructor settings so load_from can rebuild this architecture.
        checkpoint = {
            'model_state_dict': self.state_dict(),  # PyTorch state_dict maps parameter names to tensors.
            'vocab_size': self.vocab_size,
            'max_context_len': self.max_context_len,
            'embed_dim': self.embed_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        }
        torch.save(checkpoint, file_path)  # Serialize this Python dictionary with PyTorch.

    @classmethod
    def load_from(cls, file_path, device='cpu'):
        # torch.load reads the serialized checkpoint and maps tensors to the requested device.
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            ff_dim=checkpoint['ff_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )
        # Restore learned tensors by matching checkpoint names to this model's registered parameters.
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)  # Move all registered parameters/buffers to CPU or the requested accelerator.

        return model


vocab_size = 1000
max_context_len = 256
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1

# Construct a randomly initialized model; no pretrained weights or tokenizer are loaded.
model = GPT(vocab_size, max_context_len, embed_dim, n_head,
             n_layer, ff_dim, dropout_rate)

# Use random integer IDs to check the model's output dimensions, not prediction quality.
dummy_input = torch.randint(0, vocab_size, (1, max_context_len))  # PyTorch random integer IDs; shape (1,256), values 0..999.
logits = model(dummy_input)
print(f"出力形状: {logits.shape}")

# ===== CONCEPT NOTES =====
# CONCEPT: GPT adds token and position embeddings, passes their sum through causal Transformer
# blocks, normalizes, and projects every contextual vector to vocabulary logits. Logits are
# raw scores; softmax would be needed to interpret them as probabilities.
# GOAL: Assemble codebot.model.Block into a language-model forward pass, building on attention,
# LayerNorm, GELU, and FFN ideas. This is a small randomly initialized model, not pretrained GPT-2.
# INPUT: vocab_size=1000, max_context_len=256, embed_dim=384, n_head=6, n_layer=6,
# ff_dim=1536, dropout_rate=0.1. Dummy integer IDs have shape (1,256), sampled from [0,999].
# No text tokenizer or input file is involved.
# OUTPUT: Prints the output-shape label followed by torch.Size([1,256,1000]); random logits aren't printed. save()
# writes a checkpoint and load_from() reads one, but this example calls neither method.
# HOW INPUT BECOMES OUTPUT:
# 1. nn.Embedding looks up token vectors (1,256,384) and position vectors (256,384); the latter
#    broadcasts over the batch when added to token vectors.
# 2. Dropout is applied, then six causal Transformer blocks each preserve shape (1,256,384).
# 3. Final torch.nn.LayerNorm normalizes features; nn.Linear maps each 384-vector to 1,000 logits.
# NOTES: Assigning embed.weight=unembed.weight ties the lookup table and output projection.
# The example initializes weights with std=0.02 and is unseeded. C must be <= max_context_len.
# save/load store architecture settings together with state_dict; torch.load uses map_location.
