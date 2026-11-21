# Environment setup log (Ubuntu, CPU only)

Date: 2026-11-21. Project: `~/OmAiStudy/deep-learning-from-scratch-6`

## Steps

1. Checked hardware: 8 cores, 31 GB RAM, no NVIDIA GPU (`nvidia-smi` not found) -> CPU only.
2. Installed uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` -> uv 0.13.9 in `~/.local/bin`.
3. Repo already present (no clone needed). Pinned Python: `uv python pin 3.12` (`.python-version`).
4. `uv sync` was stopped: the default torch build pulls CUDA libraries (20 GB cache). Not needed without a GPU.
5. Created venv and installed CPU-only torch:
   `uv venv --python 3.12`
   `uv pip install --torch-backend cpu torch "numpy<2" matplotlib tqdm regex wandb`
6. Pinned versions to match the book:
   `uv pip install --torch-backend cpu torch==2.9.1 matplotlib==3.10.8 tqdm==4.67.1 regex==2026.1.15 wandb==0.31.0`
7. Verified: Python 3.12.3, torch 2.9.1+cpu (cuda False), numpy 1.26.4, matplotlib 3.10.8,
   tqdm 4.67.1, regex 2026.1.15, wandb 0.31.0. `ch01/01_char_tokenizer.py` runs.
8. Added `.vscode/settings.json` so VS Code uses `.venv`.

## How to run

Use the venv directly (do NOT use `uv run` / `uv sync`; they would re-install CUDA torch from the lockfile):

```
cd ~/OmAiStudy/deep-learning-from-scratch-6
source .venv/bin/activate
python ch01/01_char_tokenizer.py
```

Or without activating: `.venv/bin/python ch01/01_char_tokenizer.py`

## Differences from the macOS notes

- No MPS (Apple GPU); use CPU. Training scripts (ch03, ch05, ch06, ch09) will be slower.
- Python is 3.12.3 (system Ubuntu), not 3.12.16 from uv.
- Xcode license / Homebrew steps do not apply.

## Still to do

- Pre-trained models (codebot, storybot) via `curl` from Hugging Face, when needed (ch03+).
- Note: storybot `.pt` key names (`mlp` vs `ffn`) mismatch; handle at ch06.
- Back up files before scripts that overwrite them: `ch01/07` (codebot/merge_rules.pkl),
  `ch01/09` (codebot/tiny_codes.bin), `ch03/01`, `ch03/03`, `ch03/09` (model_*.pt).
