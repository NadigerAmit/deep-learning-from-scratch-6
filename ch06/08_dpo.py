# Concept: Direct Preference Optimization (DPO) on chosen/rejected response pairs.
# Goal: fine-tune the pretrained language model toward preferred completions.
# The reference model stays frozen while the policy model is updated with AdamW.
# The script plots loss and saves the resulting DPO checkpoint.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device

# Data, tokenizer, pretrained checkpoint, output checkpoint, and execution device.
device = get_device()
data_path = 'storybot/tiny_stories_dpo.json'
tokenizer_path = 'storybot/merge_rules.pkl'
pretrain_model_path = 'storybot/model_pretrain.pt'
dpo_model_save_path = 'storybot/model_dpo.pt'

# DPO settings: beta scales preference strength; a small LR limits policy drift.
context_len = 256
batch_size = 8
learning_rate = 5e-6
beta = 0.1
max_iters = 1000


class DPODataset(Dataset):
    # Read preference records and pre-encode each prompt/completion pair for reuse.
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            sample = self._create_sample(item['prompt'], item['chosen'], item['rejected'])
            self.samples.append(sample)

    # Build a response-only 0/1 mask; prompt and padding positions receive zero.
    def _pad_and_mask(self, ids, prompt_len):
        mask = [0] * prompt_len + [1] * (len(ids) - prompt_len)

        if len(ids) > self.context_len:
            ids = ids[:self.context_len]
            mask = mask[:self.context_len]
        else:
            pad_len = self.context_len - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len

        return ids, mask

    # Tokenize prompt and append each alternative completion to form separate sequences.
    def _create_sample(self, prompt, chosen, rejected):
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = prompt_ids + self.tokenizer.encode(chosen)
        rejected_ids = prompt_ids + self.tokenizer.encode(rejected)

        prompt_len = len(prompt_ids)
        chosen_ids, chosen_mask = self._pad_and_mask(chosen_ids, prompt_len)
        rejected_ids, rejected_mask = self._pad_and_mask(rejected_ids, prompt_len)

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    # Dataset protocol lets torch.utils.data.DataLoader batch examples into tensors.
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = self.samples[idx]
        return (
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(chosen_mask, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long),
            torch.tensor(rejected_mask, dtype=torch.long),
        )


# ids and mask are (B,C); model logits are (B,C,V), one next-token distribution per input position.
# Shift logits left and labels right so logits at t score the actual token at t+1.
def get_sequence_logprobs(model, ids, mask):
    logits = model(ids)  # (B, C, V)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    # torch.gather selects each target ID log probability: (B,C-1,V)->(B,C-1).
    per_token_logprobs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, C-1)
    # Mask prompt/padding token log-probabilities; retain completion targets only.
    masked_logprobs = per_token_logprobs * mask[:, 1:]
    return masked_logprobs.sum(dim=-1)  # (B,)


# DPO learns from preference pairs (prompt, chosen, rejected), without a separately trained reward model.
# Compare policy chosen/rejected log-ratio against frozen-reference log-ratio.
def compute_dpo_loss(model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, beta):
    # Compute policy sequence log-probabilities for chosen and rejected answers.
    chosen_logprobs = get_sequence_logprobs(model, chosen_ids, chosen_mask)
    rejected_logprobs = get_sequence_logprobs(model, rejected_ids, rejected_mask)

    # Frozen reference scores anchor the policy and limit uncontrolled drift.
    with torch.no_grad():
        ref_chosen_logprobs = get_sequence_logprobs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logprobs = get_sequence_logprobs(ref_model, rejected_ids, rejected_mask)

    # A larger relative chosen-vs-rejected log-odds makes -log(sigmoid(logit)) smaller.
    # If policy improves chosen relative to rejected by 2 log units vs reference, beta=.1 gives logit=.2.
    # F.logsigmoid is stable log(sigmoid(x)); mean reduces per-pair losses to one scalar.
    logits = beta * (
        (chosen_logprobs - rejected_logprobs) -
        (ref_chosen_logprobs - ref_rejected_logprobs)
    )
    return -F.logsigmoid(logits).mean()


# Load BPE rules, encoded examples, and shuffled batches.
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = DPODataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Policy is trainable; reference runs without gradients and stays fixed.
model = GPT.load_from(pretrain_model_path, device=device)
ref_model = GPT.load_from(pretrain_model_path, device=device)
ref_model.eval()
# AdamW updates policy parameters; the separately loaded reference model remains fixed.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Repeat preference updates; cycle reuses batches if max_iters exceeds one data pass.
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
    chosen_ids, chosen_mask = chosen_ids.to(device), chosen_mask.to(device)
    rejected_ids, rejected_mask = rejected_ids.to(device), rejected_mask.to(device)

    loss = compute_dpo_loss(
        model, ref_model,
        chosen_ids, chosen_mask,
        rejected_ids, rejected_mask,
        beta
    )

    optimizer.zero_grad()
    # Backprop computes policy gradients only; reference scores came from torch.no_grad.
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

# Matplotlib writes the loss curve; GPT.save writes weights and architecture metadata.
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("loss_dpo.png", bbox_inches='tight')

model.save(dpo_model_save_path)

# ===== CONCEPT NOTES =====
# CONCEPT: DPO directly optimizes preference pairs using policy and frozen-reference sequence log probabilities. Its scalar logit is beta*((policy_chosen-policy_rejected)-(ref_chosen-ref_rejected)); loss is -logsigmoid(logit), averaged over the batch. The response-only mask excludes prompt tokens from summed sequence log-probabilities.
# GOAL: Adapt the pretrained Storybot model to prefer chosen responses over rejected ones without training a separate reward model. Its saved checkpoint is compared against the pretraining checkpoint by ch06/09_llm_judge.py.
# INPUT: storybot/tiny_stories_dpo.json containing records with prompt, chosen, and rejected; storybot/merge_rules.pkl; initial checkpoint storybot/model_pretrain.pt. Settings: context_len=256, batch_size=8, learning_rate=5e-6, beta=0.1, max_iters=1000. Each text is BPE encoded; IDs are truncated or padded with ID 0, and masks mark completion tokens but not prompt/padding.
# OUTPUT: Prints tqdm loss values; saves the per-iteration loss plot to loss_dpo.png and the updated model to storybot/model_dpo.pt. It does not print a final numeric summary.
# HOW INPUT BECOMES OUTPUT: 1) Encode prompt and concatenate separately with chosen and rejected completion IDs. 2) Pad/truncate each to length 256; create a 0/1 completion mask. 3) For each batch, run policy logits (B,256,10000), shift logits/labels by one, gather each target token's log-probability, mask prompt positions, and sum over sequence. 4) Repeat under no_grad for the frozen reference model. 5) Apply the beta-scaled relative preference formula and -logsigmoid, backpropagate only through policy, and update it with AdamW. 6) Plot the 1000 collected loss scalars and save model weights/metadata.
# NOTES: Requires the JSON dataset, tokenizer, and compatible storybot/model_pretrain.pt. The output storybot/model_dpo.pt is overwritten on a successful run; loss_dpo.png is also overwritten in the current directory. Training and remote judging are expensive and must not be run for this task; CPU-only execution is supported but very slow. Check checkpoint architecture/state-dict keys: the imported storybot.model.GPT must match, including names such as blocks.N.ffn (a checkpoint using blocks.N.mlp will not load). Padding uses token ID 0; unusually long prompts can be truncated before the answer while the completion mask is likewise clipped.
