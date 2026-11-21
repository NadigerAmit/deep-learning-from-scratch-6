# Concept: use group-relative rewards to refine an instruction-tuned GPT on arithmetic.
# Goal: compare sampled answers within each prompt group and optimize response tokens.
# This optional alignment step starts from the SFT model and saves a GRPO checkpoint.
# The reward is exact agreement with the integer sum parsed from a generated answer.

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


# Dataset: create arithmetic prompts and exact integer answers for all 9×9 pairs.
# GRPO (Group Relative Policy Optimization) samples several answers per prompt and compares their rewards.
# Advantage = reward minus group mean: it indicates which answers did better than their peers.
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):
        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(response)

            ids = prompt_ids + response_ids
            # A 1 marks answer tokens for the policy objective; prompt tokens provide context only.
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        # Padding equalizes batch lengths; masks mark actual response positions.
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        # Batching produces IDs and masks shaped (number_of_responses,T); long IDs index the embedding table.
        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask


# Reward function: return 1 for the correct sum and 0 otherwise.
# This reward is 1 for an exact arithmetic answer and 0 otherwise; it does not grade reasoning.
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])  # Use the final integer in the answer, even if it includes an explanation.
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


# Sample multiple answers per prompt, then compare their rewards within each group.
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        # Above-average rewards give positive advantages; identical group rewards give zero learning signal.
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)

# Policy loss compares response-token probabilities from current and sampling policies.
# For IDs (B,T), select observed next-token probabilities from logits (B,T,V), yielding (B,T-1).
def compute_probs(model, ids):
    logits = model(ids)  # (B, C, V)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, C-1, V)
    labels = ids[:, 1:]  # (B, C-1)

    # torch.gather selects each observed target ID’s probability from the vocabulary dimension.
    token_probs = torch.gather(
        probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, C-1)

    return token_probs

# Clipping limits policy ratios to roughly [0.8,1.2], preventing a single update from moving too far.
def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    # Current-policy probabilities retain gradients for learning.
    probs = compute_probs(model, ids)
    # Reference probabilities come from the sampler; no graph is needed.
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    # Current/reference ratio; a tiny offset prevents division by zero.
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)

    # Broadcast one response-level advantage over tokens; the response mask later excludes prompt positions.
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]  # Shift response mask to align with next-token probabilities.
    token_objective = torch.min(unclipped, clipped) * mask

    # Normalize by sampled responses so larger groups do not automatically scale loss.
    n_samples = ids.size(0)  # batch_size × group_size
    return -token_objective.sum() / n_samples


# Runtime setup: load the SFT checkpoint/tokenizer and select a device.
device = get_device()
tokenizer_path = 'codebot/merge_rules.pkl'
sft_model_path = 'codebot/model_sft.pt'
grpo_model_save_path = 'codebot/model_grpo.pt'

# A small learning rate limits policy movement; group size controls comparisons.
learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2  # Optimizer updates per sampled group.
eval_interval = 10
epsilon = 0.2  # Clip probability ratios to [0.8,1.2].
group_size = 8  # Completions sampled per prompt.
batch_size = 32

# Load trainable policy and reference copy; synchronize the reference before sampling.
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(sft_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

old_model = GPT.load_from(sft_model_path, device=device)  # Reference policy for sampling and probability ratios.
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)

# Each iteration samples fresh responses, then updates on that batch.
accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    # Fetch prompts and their matching ground-truth sums.
    prompts, gts = next(data_iter)

    # Snapshot weights so generation and ratios use the same reference policy.
    old_model.load_state_dict(model.state_dict())

    # Sample completions and center rewards to form relative advantages.
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    # Tokenize/pad responses into IDs and response masks.
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    all_advantages = all_advantages.to(device)

    # Reuse sampled data for multiple optimizer updates instead of regenerating.
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_ rescales excessive gradient norms before AdamW updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping stabilizes updates.
        optimizer.step()

    # Evaluate all prompts greedily; evaluation does not update weights.
    if i % eval_interval == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for prompt, gt in dataset.data:
                response = generate(model, tokenizer, prompt, temperature=0)
                reward = calculate_reward(gt, response)
                correct += reward > 0
                total += 1
        model.train()
        current_accuracy = correct / total * 100
        accuracies.append(current_accuracy)

    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_accuracy:.1f}%'})

# Save the trained policy checkpoint for inference.
model.save(grpo_model_save_path)

plt.figure()
steps = list(range(0, len(accuracies) * eval_interval, eval_interval))
plt.plot(steps, accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('GRPO Training')
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_grpo.png")

# ===== CONCEPT NOTES =====
# CONCEPT: Group Relative Policy Optimization (GRPO) here ranks sampled arithmetic
# answers using binary correctness rewards and clipped token-probability ratios.
# GOAL: Refine codebot/model_sft.pt on the 81 addition prompts for i,j in 1..9.
# This optional stage follows SFT; ch03/04_chat.py can load model_grpo.pt instead.
# INPUT: codebot/merge_rules.pkl and codebot/model_sft.pt. The training settings are
# learning rate 7e-6, 500 iterations, batch size 32 prompts, 8 responses per prompt,
# 2 optimizer updates per generated batch, clip epsilon 0.2, and evaluation every
# 10 iterations. Generation uses temperature 1.0 and the utility default cap of
# 1,000 new tokens; evaluation uses greedy decoding.
# OUTPUT: tqdm displays loss and the most recent accuracy; model_grpo.pt is saved.
# loss_grpo.png plots evaluation accuracy (%) against iteration despite its filename.
# HOW INPUT BECOMES OUTPUT:
# 1. GRPODataset builds prompts and integer ground truths; each batch is sampled.
# 2. old_model is synchronized from the trainable model; generate_group samples
#    eight completions per prompt and calculate_reward extracts the last integer.
# 3. Rewards are centered within each prompt group to form advantages; get_batch
#    tokenizes prompt+response sequences, pads IDs, and marks response positions.
# 4. compute_probs gathers each next-token probability. grpo_loss compares current
#    and old probabilities, clips their ratio, masks prompt positions, and combines
#    the objective with each response's advantage.
# 5. Each batch gets two backward/gradient-clipped/AdamW updates. Every tenth step,
#    all 81 prompts are evaluated greedily; finally the model and accuracy plot save.
# NOTES: If every answer in a group has the same reward, centered advantages are
# zero for that group. The reward parser uses the last integer found in the full
# generated text. Checkpoint/plot paths are overwritten. Sampling plus full-set
# evaluation is compute-heavy on CPU; get_device can select CPU, and no GPU is
# required by this single-process script.
