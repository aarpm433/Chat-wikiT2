def reciprocal_rank_fusion(keyword_ids: list[int], vector_ids: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(keyword_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def test_rrf_boosts_shared_top_results():
    # Doc 42 appears in both keyword and vector top results
    ranked = reciprocal_rank_fusion(keyword_ids=[10, 42, 99], vector_ids=[42, 7, 10])
    assert ranked[0] == 42
    assert ranked[1] == 10
