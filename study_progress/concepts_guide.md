# Concepts Guide: the ideas behind each lesson, tied to the code

Read this with the code open. Each section says what the idea is, why it exists, and where it appears in this repo.
Shapes: `B` = batch size, `C` = number of positions (context length), `E` = embedding size, `V` = vocabulary size.

## The big picture: how an LLM works

```
text --tokenizer.encode--> token IDs --GPT model--> scores for the next token --pick one--> new token ID
 ^                                                                                              |
 +------------------------------ tokenizer.decode <-- repeat until done -----------------------+
```

1. **Tokenizer** (ch01, ch04, ch07): turns text into integers and back.
2. **Model** (ch02, ch05): a Transformer that reads token IDs and predicts the next token.
3. **Training** (ch03, ch06, ch09): adjust the model's numbers so its predictions get better.
4. **Generation** (ch03/02, ch06/06, ch09/02): use the trained model to write text.

---

## 1. Tokenizer, encode, decode (ch01/01, ch01/02)

- A neural network only does maths on numbers, so text must become numbers. A **tokenizer** does that.
- **encode(text) -> list of ints** (token IDs). **decode(ids) -> text**. They must be exact inverses.
- A **token** is one unit of text. A **vocabulary** is the table of all tokens, each with an ID. **V** is its size.
- `ch01/01_char_tokenizer.py`: `CharTokenizer.encode` uses Python's built-in `ord()` (character -> Unicode number), and `decode` uses `chr()` (number -> character). No library needed. Vocabulary: over a million possible IDs. Wasteful.
- `ch01/02_byte_tokenizer.py`: `ByteTokenizer` uses `str.encode('utf-8')` (text -> bytes) and `bytes.decode('utf-8')`. Every ID is 0 to 255, so the vocabulary is only 256 tokens. But `世` becomes 3 bytes and `😁` becomes 4, so sequences get long.
- Trade-off: small vocabulary means long sequences; large vocabulary means short sequences. BPE finds a middle path.

## 2. BPE: Byte Pair Encoding (ch01/03 to ch01/06)

- **Idea:** start from bytes, then repeatedly merge the most frequent neighbouring pair into one new token. Frequent chunks (like `th`, `ing`, ` the`) become single tokens.
- **Training** (`train_bpe` in `ch01/03_bpe_train.py`):
  1. `ids` = the text as bytes (0 to 255).
  2. `count_pairs(ids)` counts every adjacent pair, e.g. `(104, 101)` appears 7 times.
  3. Pick the most frequent pair and give it a new ID (256, then 257, ...).
  4. `merge(ids, pair, new_id)` replaces every occurrence of that pair by the new ID. The text gets shorter.
  5. Record the rule `pair -> new_id`. Repeat until the vocabulary reaches `vocab_size`.
- The list of rules is the **merge rules** (saved as `merge_rules.pkl`). Training the tokenizer is just learning these rules from text.
- **Encoding** (`BPETokenizer.encode` in `ch01/04`): turn text into bytes, then apply the merge rules in the order they were learned. **Decoding**: look up each ID's bytes and join them, then UTF-8 decode.
- **Special tokens** (`ch01/05`): reserved tokens such as `<|endoftext|>` that mark a boundary between documents. They are never split or merged.
- **Pre-tokenization** (`ch01/06`): first split text into words/chunks with a regular expression (the `regex` library), then run BPE inside each chunk. This stops merges from crossing word boundaries and makes training faster.
- ch04 and ch07 make the same BPE faster: caching, chunking, and multiprocessing. The result is the same, only quicker.

## 3. Embedding (ch02/10)

- A token ID like `5421` is just a label. The model needs a vector it can do maths on.
- An **embedding table** (`nn.Embedding(V, E)` in PyTorch) is a `V x E` matrix of learnable numbers. Row `i` is the vector for token `i`.
- Input `(B, C)` integer IDs -> output `(B, C, E)` floats. These numbers are learned during training.
- **Position embedding:** attention itself does not know order, so a vector for each position is added (GPT-2 in ch02/10). ch05 replaces this with RoPE.

## 4. Dot product and similarity (ch02/01, ch02/02)

- The **dot product** of two vectors multiplies matching entries and adds them: `(6,4,5)·(8,2,3) = 48 + 8 + 15 = 71`.
- A larger value means the vectors point in a more similar direction. So the dot product is a **similarity score**.
- A **matrix product** is many dot products at once: `A @ B` computes every row of A against every column of B.
- `ch02/01_soft_dict.py`: an ordinary dictionary finds one exact key. A **soft dictionary** compares the query to every key, turns the scores into weights, and returns a weighted average of all the values. This is exactly the idea behind attention.

## 5. Softmax (ch02/01, ch02/02, ch02/03)

- Softmax turns any list of scores into **weights that are positive and sum to 1**: `w_i = exp(s_i) / sum_j exp(s_j)`.
- Example: scores `[0, 1]` give weights about `[0.269, 0.731]`. Bigger score means bigger weight, but nothing becomes exactly 0.
- Used (a) in attention, to decide how much to look at each token, and (b) at the model's output, to turn scores into next-token probabilities.
- In PyTorch: `torch.softmax(x, dim=-1)` or `F.softmax`. `dim=-1` means "normalise along the last axis".

## 6. Attention (ch02/02 to ch02/07)

- **Goal:** let each token gather information from other tokens, weighted by relevance.
- Each token vector is projected (by learned matrices, `nn.Linear`) into three vectors:
  - **Q (query):** what am I looking for?
  - **K (key):** what do I offer?
  - **V (value):** the information I will pass on.
- **Formula:** `Attention(Q,K,V) = softmax(Q·Kᵀ / sqrt(d)) · V`
- Steps in `ch02/02_attn_math.py` (`attention(Q, K, V)`):
  1. `Q @ Kᵀ`: a `(C, C)` table of similarity scores, one for each pair of positions.
  2. Divide by `sqrt(d)` (`ch02/03_attn_scaling.py`): without this, large dot products make softmax too sharp (one weight near 1, the rest near 0) and training gets unstable.
  3. Softmax over each row: weights per position, summing to 1.
  4. `weights @ V`: each token's output is a weighted average of all the values.
- **Causal mask** (`ch02/06_attn_mask.py`): when predicting the next word, a token must not see the future. Scores for later positions are set to `-inf` before softmax, so their weights become 0. This is a lower-triangular mask.
- **Output projection** (`ch02/07_attn_value.py`): a final linear layer after the weighted average.
- **Multi-head attention** (`ch02/08_multi_head.py`, `MultiHeadAttention`): run several smaller attentions (heads) in parallel, each free to learn a different kind of relation (grammar, nearby words, etc.), then concatenate and mix them.

## 7. Transformer and GPT (ch02/09, ch02/10)

- A **Transformer block** (`Block` in `ch02/09_norm_gelu.py`) = attention + a feed-forward network (FFN), each wrapped with **LayerNorm** and a **residual connection**:
  ```
  x = x + Attention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))
  ```
- **Residual connection** (`x + ...`): the block adds a correction to its input instead of replacing it. This keeps gradients flowing in deep networks.
- **LayerNorm:** rescales each token's vector to mean 0 and variance 1, then applies learnable scale and shift. Keeps numbers stable.
- **FFN:** two linear layers with a nonlinearity between. Attention mixes information across tokens; the FFN processes each token on its own.
- **GELU:** the smooth nonlinearity used in the FFN. Without a nonlinearity, stacked linear layers would collapse into one.
- **GPT** (`GPT` in `ch02/10_gpt2.py`):
  1. Token IDs `(B, C)` -> token embeddings plus position embeddings `(B, C, E)`.
  2. Pass through `n_layer` Blocks (shape stays `(B, C, E)`).
  3. Final LayerNorm, then a linear layer maps `E -> V`: **logits** `(B, C, V)`, one score per vocabulary token at each position.
- "GPT" means a decoder-only Transformer with a causal mask that predicts the next token.

## 8. Training: pretraining, loss, optimizer (ch03/01, ch06/02, ch06/05, ch09/01)

- **Next-token prediction:** the data is a long stream of token IDs. Input is tokens `0..n-1`, target is tokens `1..n` (shifted by one).
- **Batch / block size:** cut the stream into windows of `C` tokens, and stack `B` windows into one batch.
- **Logits -> loss:** **cross-entropy** compares the predicted distribution (softmax of logits) with the true next token. It is small when the model gave the right token a high probability.
- **Backward pass** (`loss.backward()`): PyTorch computes the **gradient**, i.e. how each weight should change to reduce the loss.
- **Optimizer step** (`optimizer.step()`): change every weight a little in that direction. The **learning rate** is the step size.
- **AdamW** (ch06/02): an optimizer that keeps a running average of gradients (momentum) and their size (per-weight step scaling), plus **weight decay** to keep weights small.
- **LR schedule** (ch06/03): learning rate rises during **warmup**, then decays (cosine). Stabilises training.
- **Mixed precision** (ch06/04): use 16-bit floats (fp16/bf16) to save memory and time. Mostly useful on GPUs; on CPU the gain is small.
- **Checkpoint:** saving the weights to a file (`model_pretrain.pt`) so you can reuse them.
- On this CPU-only machine, training scripts are slow. Pretrained checkpoints can be downloaded instead.

## 9. Generation and sampling (ch03/02, ch06/06, ch09/02)

- Loop: run the model on the tokens so far, take the logits of the **last position**, softmax them into probabilities, **sample** one token, append it, repeat.
- **Temperature:** divide logits by T before softmax. Low T makes it more deterministic; high T makes it more random.
- `temperature == 0` in `ch03/02_generate.py` means always pick the most likely token (greedy); otherwise the code samples from `F.softmax(logits / temperature, dim=-1)`.
- (General background, not used in these scripts: **top-k** sampling restricts the choice to the k most likely tokens.)
- Stop when the model produces the end token or reaches `max_new_tokens`.

## 10. SFT and chat (ch03/03, ch03/04)

- A pretrained model only continues text. **SFT (supervised fine-tuning)** trains it on (instruction, answer) pairs so that it follows instructions.
- A **chat template** wraps text in role markers (e.g. a user part and an assistant part). The loss is typically computed only on the answer tokens.
- `ch03/04_chat.py` runs a conversation loop on the fine-tuned model.

## 11. Reinforcement learning: GRPO (ch03/09)

- After SFT, the model is improved with a **reward** (a score for how good an answer is, e.g. whether code passes tests).
- **GRPO:** sample several answers for one prompt, score them, and compute each answer's **advantage** = its reward compared with the group average. Raise the probability of above-average answers and lower it for below-average ones.

## 12. Modern improvements (ch05)

| Idea | File | What it replaces / does |
|---|---|---|
| **RoPE** (rotary position embedding) | `ch05/01_rope.py` | Instead of adding position vectors, rotate pairs of numbers in Q and K by an angle that depends on position. Then Q·K depends on the relative distance of tokens. |
| **SwiGLU** | `ch05/02_swiglu.py` | A gated FFN: `a = W(x)`, `b = V(x)`, `out = O(silu(a) * b)`. `silu(x) = x * sigmoid(x)`. Replaces the GELU FFN. |
| **RMSNorm** | `ch05/03_rmsnorm.py` | A simpler LayerNorm: divides by the root-mean-square only (no mean subtraction). Cheaper. |
| **Improved GPT** | `ch05/04_improved_gpt.py` | The GPT from ch02 with RoPE + SwiGLU + RMSNorm. This is the style of Llama-type models. |
| **KV cache** | `ch05/05_kvcache.py` | During generation, K and V of earlier tokens never change, so store them. Each new token then needs only one new row of attention instead of recomputing everything. |

## 13. Preference learning and judging (ch06/07 to ch06/09)

- **DPO** (`ch06/08_dpo.py`): train on pairs (chosen answer, rejected answer). The loss pushes the model to prefer the chosen one more than a frozen reference model does, without a separate reward model.
- **LLM-as-judge** (`ch06/07_llm_judge.py`, `ch06/09_llm_judge.py`): ask another LLM to score or compare answers. Both scripts create `OpenAI(api_key="your_api_key_here")`, so they need a real OpenAI API key and internet access.

---

## Chapter map

| Chapter | Topic | Key concepts |
|---|---|---|
| ch01 | Tokenizers | encode/decode, bytes, BPE, special tokens, pre-tokenization |
| ch02 | Attention and GPT | dot product, softmax, Q/K/V, mask, multi-head, LayerNorm, GELU, Block, GPT |
| ch03 | Train and use a small GPT | pretraining, generation, SFT, chat, GRPO |
| ch04 | Fast BPE | caching, chunking, parallelism |
| ch05 | Modern GPT | RoPE, SwiGLU, RMSNorm, KV cache |
| ch06 | Training techniques | AdamW, LR schedule, mixed precision, DPO, LLM judge |
| ch07 | BPE on a web-text corpus | trains and applies BPE with the fast parallel code from ch04 |
| ch09 | Larger GPT on OpenWebText-style data | pretraining with RoPE, grouped-query attention and SwiGLU, then generation |

## Python libraries used

| Library | Used for |
|---|---|
| built-ins (`ord`, `chr`, `str.encode`, `bytes.decode`) | the simple tokenizers in ch01 |
| `regex` | Unicode-aware splitting of text before BPE |
| `collections` (`Counter`, `defaultdict`) | counting pairs in BPE |
| `pickle` | saving and loading merge rules |
| `numpy` | plain-array maths in early examples (ch02/01) |
| `torch`, `torch.nn`, `torch.nn.functional` | tensors, layers (`nn.Linear`, `nn.Embedding`), softmax, loss |
| `torch.optim` | optimizers such as AdamW |
| `tqdm` | progress bars |
| `matplotlib` | plots (attention weights, learning-rate curves) |
| `wandb` | optional experiment tracking |
| `openai` | LLM-as-judge in ch06 |
