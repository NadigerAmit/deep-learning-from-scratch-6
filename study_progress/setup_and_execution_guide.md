# Setup and Execution Guide (Ubuntu, CPU only)

Project: `~/OmAiStudy/deep-learning-from-scratch-6` (book: Deep Learning from Scratch 6, LLM edition)

## 1. Setup steps (done 2026-11-21)

| # | Step | Command / result |
|---|---|---|
| 1 | Check hardware | 8 cores, 31 GB RAM, no NVIDIA GPU (`nvidia-smi` not found) -> CPU only |
| 2 | Install uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` then `source "$HOME/.local/bin/env"` -> uv 0.13.9 |
| 3 | Repo | Already present; no clone needed |
| 4 | Pin Python | `uv python pin 3.12` -> `.python-version` |
| 5 | First attempt | `uv sync` stopped: default torch pulls CUDA libraries (20 GB cache), useless without a GPU |
| 6 | Clean up | `uv cache clean; rm -rf .venv` |
| 7 | Create venv | `uv venv --python 3.12` (uses system Python 3.12.3) |
| 8 | Install CPU torch + libs | `uv pip install --torch-backend cpu torch "numpy<2" matplotlib tqdm regex wandb` |
| 9 | Pin book versions | `uv pip install --torch-backend cpu torch==2.9.1 matplotlib==3.10.8 tqdm==4.67.1 regex==2026.1.15 wandb==0.31.0` |
| 10 | VS Code | `.vscode/settings.json` points to `.venv/bin/python` |
| 11 | Smoke test | `python ch01/01_char_tokenizer.py` printed the expected IDs and `hello世界😁` |

Installed: Python 3.12.3, torch 2.9.1+cpu, numpy 1.26.4, matplotlib 3.10.8, tqdm 4.67.1, regex 2026.1.15, wandb 0.31.0.

## 2. How to run code

```bash
cd ~/OmAiStudy/deep-learning-from-scratch-6
source .venv/bin/activate
python ch01/01_char_tokenizer.py
```

Without activating: `.venv/bin/python ch01/01_char_tokenizer.py`

Do NOT use plain `uv run` or `uv sync`: they re-resolve from `uv.lock` and install the CUDA build of torch.
Use `uv run --no-sync` instead; it uses the existing `.venv` as is:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python ch01/01_char_tokenizer.py
uv run --no-sync --with tiktoken python ch01/01_char_tokenizer.py   # temporary extra library
```

### One-command lessons (`lesson.sh`)

Copilot CLI (`@github/copilot`) is installed in `~/.local/bin` (via npm, no sudo). First run: `copilot` then `/login`.

```bash
cd ~/OmAiStudy/deep-learning-from-scratch-6
./lesson.sh ch02 03                       # guided: Step 1 explain -> Step 2 run -> Step 3 explain + record progress
./lesson.sh --run ch02 03                 # run only
./lesson.sh --run --with tiktoken ch02 03 # run only, with a temporary library
./lesson.sh list 2                        # list lessons in a chapter
```

Guided mode starts `copilot` in `~/OmAiStudy` with the `/lesson` skill and asks "Shall we move on?" after each step.
Run-only mode backs up files that a lesson overwrites (`*_original.*`) first.

## 3. How to verify the installation

Quick check of every library and version:

```bash
source .venv/bin/activate
python - <<'EOF'
import sys, torch, numpy, matplotlib, tqdm, regex, wandb
print("python    ", sys.version.split()[0])
print("torch     ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy     ", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("tqdm      ", tqdm.__version__)
print("regex     ", regex.__version__)
print("wandb     ", wandb.__version__)
print("torch test", (torch.ones(2, 2) @ torch.ones(2, 2)).tolist())
EOF
```

Expected: versions as listed above, `cuda: False`, and `torch test [[2.0, 2.0], [2.0, 2.0]]`.

Other checks:

```bash
uv pip list | grep -Ei "torch|numpy|matplotlib|tqdm|regex|wandb"   # installed versions
which python                                                       # should end with .venv/bin/python
python ch01/01_char_tokenizer.py                                   # end-to-end smoke test
```

## 4. Why each library is useful

| Library | Role in this book |
|---|---|
| Python 3.12 | Language; the book needs 3.10+. |
| torch (PyTorch) | Tensors and automatic differentiation. Builds, trains and runs the GPT model (attention, training loops, generation). CPU build here, so training is slower. |
| numpy | Array maths for the early, from-scratch examples (e.g. attention maths) and for storing token data (`.bin` files). Pinned below 2.0 for compatibility. |
| matplotlib | Plots: attention weights, learning-rate schedules, loss curves. |
| tqdm | Progress bars for long loops such as BPE training and model training. |
| regex | Unicode-aware regular expressions used to pre-tokenize text before BPE (ch01). |
| wandb | Optional experiment tracking (loss curves, runs) during training. Can run offline or be disabled. |
| uv | Fast Python/package manager that created the venv. |

## 5. Execution log of lessons

| Date | Lesson | Status | Notes |
|---|---|---|---|
| 2026-11-21 | ch01/01_char_tokenizer.py | Explained and run | `ord`/`chr` char tokenizer; output `[104, 101, 108, 108, 111, 19990, 30028, 128513]` and round-trip decode OK |

## 6. Cautions

- No Apple MPS here; scripts run on CPU. Training in ch03, ch05, ch06, ch09 will be slow.
- Back up before scripts that overwrite files: `ch01/07` (`codebot/merge_rules.pkl`), `ch01/09` (`codebot/tiny_codes.bin`), `ch03/01`, `ch03/03`, `ch03/09` (`model_*.pt`).
- Pre-trained models (codebot, storybot) are downloaded from Hugging Face with `curl` when needed (ch03+). Storybot `.pt` key names (`mlp` vs `ffn`) mismatch; handle at ch06.
