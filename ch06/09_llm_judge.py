# Concept: pairwise LLM-as-a-Judge evaluation of pretrained and DPO stories.
# Goal: estimate whether preference tuning yields happier endings than the baseline.
# Two stories are generated per comparison and randomly swapped to reduce position bias.
# An external judge selects A, B, or tie; aggregate wins are printed.

import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import torch
import torch.nn.functional as F
from openai import OpenAI
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate

# API client, device, and tokenizer configuration.
# ==========================================
client = OpenAI(api_key="your_api_key_here")
# ==========================================
device = get_device()
tokenizer_path = 'storybot/merge_rules.pkl'
tokenizer = BPETokenizer.load_from(tokenizer_path)

# Compare baseline pretraining against the preference-tuned checkpoint.
model_paths = {
    'pretrain': 'storybot/model_pretrain.pt',
    'dpo': 'storybot/model_dpo.pt',
}

# Pairwise evaluation uses the same prompt and generation controls for both models.
prompt = "Once upon a time"
num_comparisons = 100  # Number of pairwise comparisons.
max_new_tokens = 150
temperature = 1.0


def compare_stories(client, story_a, story_b):
    """Compare two stories and choose the happier or more hopeful ending."""

    # The Japanese functional prompt asks which children’s story has a happier/hopeful ending; preserve as API input.
    evaluation_prompt = f"""以下の2つの子供向けストーリーを比較し、どちらがよりハッピーエンドかを判定してください。

【Story A】
{story_a}

【Story B】
{story_b}

どちらがより明るく幸せな結末か、または希望に満ちた内容かを判断してください。
JSON形式で回答: {{"winner": "A" or "B" or "tie", "reason": "簡潔な理由"}}"""

    # OpenAI chat completion returns a JSON winner label and reason for this pair.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=150,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    return json.loads(text)

# Load both local checkpoints once, then reuse them in all comparisons.
model_pretrain = GPT.load_from(model_paths['pretrain'], device=device)
model_dpo = GPT.load_from(model_paths['dpo'], device=device)

# Keep detailed outcomes in memory and counts by model name or tie.
results = []
wins = {"pretrain": 0, "dpo": 0, "tie": 0}

# 100 comparisons mean 200 local generations plus 100 remote judge requests; outputs are not logged to W&B.
for i in range(num_comparisons):
    print(f"\n{'='*60}")
    print(f"Comparison {i+1}/{num_comparisons}")
    print('='*60)

    # Generate one sample from each model before randomizing A/B labels.
    story_pretrain = generate(model_pretrain, tokenizer, prompt, max_new_tokens, temperature)
    story_dpo = generate(model_dpo, tokenizer, prompt, max_new_tokens, temperature)

    print(f"\n[Pretrain]: {story_pretrain[:100]}...")
    print(f"\n[DPO]: {story_dpo[:100]}...")

    # Randomize A/B order so judge slot bias is less confounded with model identity.
    import random
    if random.random() < 0.5:
        story_a, story_b = story_pretrain, story_dpo
        mapping = {"A": "pretrain", "B": "dpo"}
    else:
        story_a, story_b = story_dpo, story_pretrain
        mapping = {"A": "dpo", "B": "pretrain"}

    # Ask an external OpenAI judge to select A, B, or tie; API use may incur cost.
    judgment = compare_stories(client, story_a, story_b)

    winner_label = judgment["winner"]
    if winner_label == "tie":
        winner = "tie"
    else:
        winner = mapping[winner_label]

    wins[winner] += 1

    print(f"\n🏆 Winner: {winner}")
    print(f"   Reason: {judgment['reason']}")

    results.append({
        "story_pretrain": story_pretrain,
        "story_dpo": story_dpo,
        "winner": winner,
        "reason": judgment["reason"]
    })

# Convert counts to percentages; ties remain in the denominator here.
print("\n" + "="*60)
print("📊 PAIRWISE COMPARISON RESULTS")
print("="*60)

total = num_comparisons
print(f"\n  Pretrain wins: {wins['pretrain']:3d} ({wins['pretrain']/total*100:5.1f}%)")
print(f"  DPO wins:      {wins['dpo']:3d} ({wins['dpo']/total*100:5.1f}%)")
print(f"  Ties:          {wins['tie']:3d} ({wins['tie']/total*100:5.1f}%)")

# Compute DPO win rate only among decisive non-tie judgments.
if wins['pretrain'] + wins['dpo'] > 0:
    dpo_winrate = wins['dpo'] / (wins['pretrain'] + wins['dpo']) * 100
    print(f"\n  DPO win rate (excluding ties): {dpo_winrate:.1f}%")

# ===== CONCEPT NOTES =====
# CONCEPT: Pairwise LLM-as-a-Judge evaluation compares two candidate outputs against a rubric (happier/more hopeful ending). Randomly swapping A/B assignments reduces systematic position bias; labels are mapped back to model names before counting wins.
# GOAL: Compare the pretrained model with the DPO-tuned model after training, complementing the single-story coherence/grammar evaluation in ch06/07_llm_judge.py.
# INPUT: storybot/merge_rules.pkl, storybot/model_pretrain.pt, and storybot/model_dpo.pt; prompt `Once upon a time`; num_comparisons=100; max_new_tokens=150; temperature=1.0. The API client uses the placeholder `your_api_key_here`; judge is gpt-4o-mini with JSON response mode and max_tokens=150.
# OUTPUT: Prints truncated previews of both stories, each judgment and reason, then counts and percentages for pretrained wins, DPO wins, and ties. It also prints DPO win rate excluding ties if there is at least one non-tie. It does not save results; it generates 200 stories and makes 100 judge calls.
# HOW INPUT BECOMES OUTPUT: 1) Load both checkpoints and tokenizer. 2) For each of 100 comparisons, independently sample one story from each model (up to 150 new tokens). 3) Randomly assign the two stories to A/B and remember the reverse mapping. 4) Ask the judge for JSON winner A, B, or tie plus a reason. 5) Map the winner to pretrain/DPO, increment counts, and compute percentages over 100; calculate DPO's non-tie win fraction when possible.
# NOTES: Requires the DPO checkpoint created by ch06/08_dpo.py plus baseline and tokenizer files; a missing storybot/model_dpo.pt stops execution. Replace the API-key placeholder with a valid key; network/API cost applies and results vary. Local generation runs on CPU but can be slow; the judge runs remotely. The script only prints results and keeps the detailed list in memory.
