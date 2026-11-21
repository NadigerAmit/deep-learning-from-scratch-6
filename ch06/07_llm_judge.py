# Concept: score generated stories with an external LLM-as-a-Judge.
# Goal: compare coherence and grammar across pretraining checkpoints.
# Each checkpoint generates ten samples, and GPT-4o mini returns JSON scores.
# Results are summarized in the console; this script does not save a report.

import os
import sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
import statistics
from openai import OpenAI
from storybot.model import GPT
from storybot.tokenizer import BPETokenizer
from storybot.utils import get_device, generate


# API client and model-evaluation configuration.
# ==========================================
client = OpenAI(api_key="your_api_key_here")
# ==========================================
device = get_device()
tokenizer_path = 'storybot/merge_rules.pkl'
tokenizer = BPETokenizer.load_from(tokenizer_path)

# These are checkpoints saved at selected training iterations.
model_paths = {
    500: 'storybot/model_iter_500.pt',
    5000: 'storybot/model_iter_5000.pt',
    40000: 'storybot/model_pretrain.pt',
}

# Generation settings apply to each sample from each checkpoint.
prompt = "<|endoftext|>"
max_new_tokens = 200
temperature = 1.0
num_samples = 10  # Generate ten samples from each checkpoint.

def evaluate_story(client, story):
    """Ask an external language model to score one story against a rubric."""

    # The Japanese functional prompt asks for coherence (logical story flow) and grammar (correct English).
    # It requests JSON fields coherence, grammar, and a short comment; this string is API data, not a comment.
    evaluation_prompt = f"""以下の子供向けストーリーを2つの観点で1-5点で評価してください。

ストーリー:
{story}

評価観点:
1. Coherence（一貫性）: 論理的につながっているか、物語として筋が通っているか
2. Grammar（文法）: 文法的に正しい英語か

以下のJSON形式で回答してください:
{{
    "coherence": <1-5の整数>,
    "grammar": <1-5の整数>,
    "comment": "<評価の簡単な理由>"
}}"""

    # openai.OpenAI chat.completions.create sends one user message and returns a response object.
    # response_format requests JSON; network, API, and response-schema failures remain possible.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    # The Japanese console heading means “output.”
    print("=====出力====")
    print(text)

    # JSON response mode requests machine-readable output, simplifying json.loads parsing.
    return json.loads(text)

# This loop generates 3 checkpoints * 10 stories = 30 stories and makes 30 remote judge requests.
# No wandb client is imported: results are kept in memory and printed.
results = {}
for iteration, model_path in model_paths.items():
    print(f"\n{'='*50}")
    print(f"Iteration {iteration}")
    print('='*50)

    model = GPT.load_from(model_path, device=device)
    iteration_results = []

    for i in range(num_samples):
        # Japanese console label means “Sample N.”
        print(f"\n--- サンプル {i+1} ---")

        # Generate text locally; the external judge scores this text but does not update GPT.
        story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
        print(f"Story: {story[:200]}...")

        # OpenAI API sends the rubric and story to a separate judge model; usage may incur cost.
        scores = evaluate_story(client, story)
        print(f"Scores: {scores}")

        iteration_results.append({
            "story": story,
            "scores": scores
        })

    results[iteration] = iteration_results

# Summarize judge scores with arithmetic mean and sample standard deviation.
print("\n" + "="*50)
print("Summary")
print("="*50)

for iteration in model_paths.keys():
    scores_list = [r["scores"] for r in results[iteration]]

    print(f"\nIteration {iteration}:")
    for key in ["coherence", "grammar"]:
        values = [s[key] for s in scores_list]
        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {key}: {avg:.2f} ± {std:.2f}")

# ===== CONCEPT NOTES =====
# CONCEPT: LLM-as-a-Judge uses a separate language model to evaluate generated text against rubric criteria. Here the judge assigns 1-5 scores for coherence and grammar and returns a short comment in JSON; the script summarizes mean and sample standard deviation.
# GOAL: Compare story quality at saved pretraining stages, connecting the training checkpoints to an external qualitative evaluation. This is a separate API-based evaluator, not part of GPT training.
# INPUT: Tokenizer file storybot/merge_rules.pkl; model checkpoints at iterations 500 (storybot/model_iter_500.pt), 5000 (storybot/model_iter_5000.pt), and 40000 (storybot/model_pretrain.pt); prompt `<|endoftext|>`, max_new_tokens=200, temperature=1.0, num_samples=10 per checkpoint. OpenAI client is configured with placeholder `your_api_key_here`; judge model is gpt-4o-mini, max_tokens=300, JSON response mode.
# OUTPUT: Prints each judge JSON response and scores, then for every checkpoint prints average ± sample standard deviation for coherence and grammar. It does not write result files. It generates 30 stories total and requests one judge response for each.
# HOW INPUT BECOMES OUTPUT: 1) Load the tokenizer and each checkpoint. 2) For each of 10 samples per checkpoint, generate up to 200 tokens from the prompt. 3) Send the story embedded in the Japanese rubric prompt to gpt-4o-mini. 4) Parse its JSON with coherence, grammar, and comment; collect each response. 5) Compute statistics.mean and statistics.stdev for the two score lists and print them.
# NOTES: Requires the three checkpoints, tokenizer file, network access, and a valid OpenAI API key replacing the placeholder. API calls may incur cost and model outputs are nondeterministic. CPU can run local generation but may be slow; judging itself is remote. No plot or checkpoint is written by this script.
