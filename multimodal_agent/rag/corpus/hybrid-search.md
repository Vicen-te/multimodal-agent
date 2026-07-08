# Hybrid search

Hybrid search combines two retrieval strategies that fail in different ways, so
together they cover each other's blind spots. Dense semantic search embeds the
query and documents into vectors and ranks by cosine similarity; it captures
meaning and handles paraphrases, but it can miss exact terms such as a rare
function name or an error code. Sparse lexical search, classically BM25, ranks by
term overlap; it nails exact keywords but is blind to synonyms and rephrasing.

A hybrid retriever runs both searches over the same corpus, takes the top
candidates from each, and fuses the two rankings into a single ordered list. The
fused list is then truncated to the few chunks that are actually handed to the
language model. The usual failure mode of a pure semantic system, returning
plausible-but-wrong neighbours for a keyword-heavy query, is exactly what the
lexical half repairs, and vice versa.

The fusion step needs care because BM25 scores and cosine similarities live on
different scales and cannot be added directly. Reciprocal rank fusion sidesteps
this by ignoring the raw scores and combining ranks instead, which makes it a
robust default for production hybrid search.
