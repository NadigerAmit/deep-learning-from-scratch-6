# Concept: a hard dictionary lookup returns one value only when a key matches exactly;
# a soft dictionary compares a query with every key and blends their associated values.
# Here, each 3-number vector means (action, drama, comedy), and each value is a rating.
# The dot products are similarity scores, not probabilities or ratings themselves.
# NumPy computes the scores and weighted average; this example trains no model and reads no files.

import numpy as np

# A tiny ordinary Python dictionary: keys identify items and values are looked up exactly.
d = {
    'apple': 100,
    'banana': 200,
    'cherry': 300,
    'durian': 400
}

# Exact lookup: because 'banana' is a present key, Python returns its stored value 200.
query = 'banana'
print(d[query])  # 200


# Each tuple key is a movie's (action, drama, comedy) feature vector, with each feature from 0 to 10.
# Each value is the user's rating from 0 to 100; similar feature vectors should suggest similar ratings.
movie_preferences = {
    (8, 2, 3): 85,  # An action-heavy movie.
    (3, 9, 1): 70,  # A drama-heavy movie.
    (1, 2, 9): 60,  # A comedy-heavy movie.
    (5, 5, 5): 75,  # A balanced movie.
    (7, 6, 2): 80,  # An action drama.
    (2, 7, 6): 65,  # A comedy drama.
    (9, 1, 1): 90,  # Pure action.
}

# This is a new movie query, not one of the dictionary keys; its rating will be estimated.
new_movie = (6, 4, 5)

def soft_dictionary(query, dictionary):
    # `query` and each tuple key have shape (3,). NumPy's np.dot multiplies
    # matching coordinates and sums them to one scalar score per key. For example,
    # (6,4,5) dot (8,2,3) = 6*8 + 4*2 + 5*3 = 48+8+15 = 71.
    # A larger score means greater alignment under this dot-product measure.
    similarity = []
    for key in dictionary:
        s = np.dot(query, key)  # NumPy np.dot: shape (3,) by (3,) -> one scalar.
        similarity.append(s)  # One score per dictionary key, preserving dictionary order.

    # Softmax converts the seven scores into positive weights that sum to 1:
    # w_i = exp(s_i) / sum_j exp(s_j). For scores [0, 1], the weights are about
    # [0.269, 0.731]; the larger score gets more influence, but neither item is discarded.
    # These are mixture proportions, so their sum of 1 makes the result a weighted average.
    exp_similarity = np.exp(similarity)  # NumPy elementwise exponential: shape (7,).
    weights = exp_similarity / np.sum(exp_similarity)  # Normalize all seven weights together.

    # Blend the seven stored ratings with the corresponding weights.
    result = 0
    for weight, value in zip(weights, dictionary.values()):
        result += weight * value  # Scalar contribution; sum all contributions into one rating.

    return result, weights


predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)

print(f"新しい映画 {new_movie} の予測評価: {predicted_rating:.2f} 点")
print("\n各映画の重み:")
for key, weight in zip(movie_preferences.keys(), weights):
    print(f"映画 {key}: {weight*100:.2f}%")

# ===== CONCEPT NOTES =====
# CONCEPT: A hard Python dictionary requires an exact key match; this soft lookup compares
# all keys, assigns each one a softmax weight, and returns a weighted average of their values.
# For the query (6,4,5), its dot product with key (8,2,3) is 6*8+4*2+5*3=71.
# GOAL: Connect a familiar exact lookup (banana -> 200) to attention-like retrieval and
# prepare for the explicit query/key/value matrix calculation in 02_attn_math.py.
# INPUT: The ordinary dictionary stores four fruit/value pairs. The movie dictionary stores
# seven keys of shape (3,) and ratings [85,70,60,75,80,65,90]; `new_movie` is (6,4,5).
# OUTPUT: Prints exact value 200, then one weighted predicted rating and seven percentages.
# No file is read or written. Softmax weights sum to 1, so the predicted rating is between
# the smallest and largest stored ratings (60 and 90).
# HOW INPUT BECOMES OUTPUT:
# 1. NumPy np.dot compares the query and each key, yielding seven scalar similarity scores.
# 2. NumPy exp and sum implement w_i=exp(s_i)/sum_j exp(s_j); these positive proportions sum to 1.
# 3. Multiply each weight by its stored rating and sum to get the weighted-average prediction.
# 4. Print the result to two decimals and each weight times 100 as a percentage.
# NOTES: This is not a trained neural network. Direct np.exp can overflow for very large
# scores; stable softmax normally subtracts the maximum score first, but that is not done here.
# Python preserves dictionary insertion order, so scores, weights, keys, and ratings align.
