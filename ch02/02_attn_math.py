# Concept: attention lets each query retrieve a mixture of values by scoring every key.
# Q (queries) asks what information is relevant; K (keys) describes what each item offers;
# V (values) is the information actually blended. Scores are similarities, not probabilities.
# This batch calculation uses PyTorch tensors, but has no learned projections, scaling, or mask yet.

# torch provides tensor/matrix operations; torch.nn.functional (aliased F) provides stateless activations.
import torch
import torch.nn.functional as F

# Keys: seven movie feature vectors; tensor shape (7,3) = seven items by three genres.
# torch.tensor converts nested Python lists into a dense tensor with rows as separate keys.
K = torch.tensor([
    [8, 2, 3],  # Action-heavy movie.
    [3, 9, 1],  # Drama-heavy movie.
    [1, 2, 9],  # Comedy-heavy movie.
    [5, 5, 5],  # Balanced movie.
    [7, 6, 2],  # Action drama.
    [2, 7, 6],  # Comedy drama.
    [9, 1, 1],  # Pure action.
], dtype=torch.float32)

# Values: the rating for each key, shape (7,1); each row corresponds to the same-index key.
# Create the rating values as a column tensor so matrix multiplication returns one rating per query.
V = torch.tensor([
    [85],
    [70],
    [60],
    [75],
    [80],
    [65],
    [90]
], dtype=torch.float32)

# Three query movies, each a row of (action, drama, comedy), shape (3,3).
# Create three query rows; each row will be compared against all seven key rows.
Q = torch.tensor([
    [6, 4, 5],  # Balanced movie with a slight action emphasis.
    [2, 8, 3],  # Drama-heavy movie.
    [4, 3, 7],  # Comedy-leaning movie.
], dtype=torch.float32)

def attention(Q, K, V):
    # torch.matmul multiplies the query matrix by transposed keys: (3,3) @ (3,7) -> (3,7).
    # K.t() swaps a 2-D tensor's axes so each query gets one score for each of seven keys.
    # Example: first query (6,4,5) against first key (8,2,3) scores 48+8+15=71.
    similarity = torch.matmul(Q, K.t())  # Raw query-key scores; larger means a stronger match.
    # torch.nn.functional.softmax normalizes each query's seven scores into weights summing to 1.
    # dim=1 means normalize across the seven keys (not across the three queries).
    weights = F.softmax(similarity, dim=1)
    # Weighted retrieval: (3,7) @ (7,1) -> (3,1), one weighted rating per query.
    output = torch.matmul(weights, V)
    return output, weights

predicted_ratings, weights = attention(Q, K, V)

# Each loop iteration pairs one original query row with its one predicted scalar rating.
# Iterating over each row keeps one query paired with its one-element output.
for movie, rating in zip(Q, predicted_ratings):
    print(f"映画 {movie.numpy()} の予測評価: {rating.item():.2f}")  # Tensor.numpy converts CPU values for display; item() extracts one scalar.

# ===== CONCEPT NOTES =====
# CONCEPT: For each query, attention calculates key similarities, softmax turns them into
# nonnegative proportions summing to 1, and those proportions blend the corresponding values.
# A score is not a probability: only after softmax does each row become a probability-like distribution.
# GOAL: Express 01_soft_dict.py's soft retrieval as batched PyTorch matrix multiplication.
# Later examples add learned projections, score scaling, and causal masking.
# INPUT: K=(7,3) holds seven genre vectors; V=(7,1) holds ratings [85,70,60,75,80,65,90];
# Q=(3,3) contains (6,4,5), (2,8,3), and (4,3,7). All are torch.float32.
# OUTPUT: weights has shape (3,7), predicted_ratings has shape (3,1), and three ratings print.
# HOW INPUT BECOMES OUTPUT:
# 1. K.t() changes (7,3) to (3,7); torch.matmul(Q,K.t()) gives scores (3,7).
# 2. F.softmax(scores,dim=1) normalizes each row over seven keys; for scores [0,1], softmax is about [0.269,0.731].
# 3. torch.matmul(weights,V) computes (3,7) @ (7,1) -> (3,1), the weighted rating for each query.
# 4. The print loop displays each query and its scalar output rounded to two decimals.
# NOTES: F is torch.nn.functional; torch.matmul and Tensor.t are PyTorch operations.
# This introductory form is unscaled and unmasked, and no model parameters are learned.
