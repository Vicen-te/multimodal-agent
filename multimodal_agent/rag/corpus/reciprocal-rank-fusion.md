# Reciprocal rank fusion

Reciprocal rank fusion, or RRF, merges several ranked lists into one without
needing the underlying scores to be comparable. For every list, an item at rank
position r contributes a score of 1 divided by (k plus r), where k is a small
constant, commonly 60. An item's final score is the sum of these contributions
across all the lists it appears in, and the items are sorted by that sum.

The reason RRF works so well is that it only looks at positions, not raw scores.
A BM25 score might range into the tens while a cosine similarity sits between
minus one and one; adding them directly would let BM25 dominate purely because of
scale. By converting both to reciprocal ranks first, each ranker gets a fair
vote. The constant k dampens the influence of the very top ranks: with k around
60, the gap between rank one and rank two is gentle, so a document needs support
from more than one ranker to climb to the top.

RRF is cheap, has a single intuitive parameter, and requires no training or score
normalisation, which is why it is a standard choice for fusing the outputs of a
semantic and a lexical retriever in a hybrid search pipeline.
