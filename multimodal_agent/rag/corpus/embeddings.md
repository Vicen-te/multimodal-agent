# Sentence embeddings

A sentence embedding maps a piece of text to a fixed-length vector so that texts
with similar meaning land close together in the vector space. Retrieval then
becomes a nearest-neighbour search: embed the query, compare it to the stored
document vectors with cosine similarity, and return the closest ones.

The all-MiniLM-L6-v2 model from the Sentence Transformers library is a popular
default for this job. It produces 384-dimensional vectors, runs comfortably on a
CPU, and offers a strong balance of speed and quality for general-purpose
retrieval. Because the vectors are small, similarity search stays fast even
without a specialised vector database.

It helps to normalise embeddings to unit length before storing them. Once every
vector has length one, cosine similarity reduces to a plain dot product, which is
faster and numerically simpler. The same embedding model must be used for both
indexing the documents and embedding the queries; mixing models would place
queries and documents in incompatible spaces and wreck retrieval quality.
