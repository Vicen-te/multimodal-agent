from multimodal_agent.rag.retriever import HybridRetriever, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    # item 2 is top in neither list but appears high in both -> should win.
    semantic = [1, 2, 3]
    lexical = [4, 2, 5]
    fused = reciprocal_rank_fusion([semantic, lexical], k=60)
    ranked = [item for item, _ in fused]
    assert ranked[0] == 2


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([[7]], k=60))
    assert fused[7] == 1.0 / 60


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_retriever_dedupes_by_parent(retriever):
    results = retriever.search("graph nodes cycles")
    parent_ids = [chunk.parent_id for chunk in results]
    assert len(parent_ids) == len(set(parent_ids))
    assert len(results) <= retriever.top_k


def test_retriever_empty_store_returns_empty(embedder):
    from multimodal_agent.rag.store import HybridStore

    empty = HybridRetriever(HybridStore(), embedder)
    assert empty.search("anything") == []
