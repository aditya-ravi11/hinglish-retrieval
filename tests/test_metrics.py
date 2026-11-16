import math

from hinglish_retrieval.eval.metrics import mean, ndcg_at_k, paired_bootstrap, recall_at_k


def test_ndcg_perfect_and_hand_computed():
    qrels = {"q": {"a": 1, "b": 1}}
    assert ndcg_at_k({"q": {"a": 3, "b": 2, "c": 1}}, qrels)["q"] == 1.0
    # relevant docs at ranks 2 and 3
    got = ndcg_at_k({"q": {"c": 3, "a": 2, "b": 1}}, qrels)["q"]
    want = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert abs(got - want) < 1e-12


def test_ndcg_cutoff_and_missing_run():
    qrels = {"q": {"a": 1}, "r": {"x": 1}}
    run = {"q": {f"d{i}": 100 - i for i in range(10)} | {"a": 0}}
    res = ndcg_at_k(run, qrels, k=10)
    assert res["q"] == 0.0  # relevant doc is rank 11
    assert res["r"] == 0.0  # no run for this query counts as zero


def test_ties_broken_by_doc_id():
    qrels = {"q": {"b": 1}}
    assert ndcg_at_k({"q": {"a": 1.0, "b": 1.0}}, qrels, k=1)["q"] == 0.0


def test_recall_and_skip_queries_without_positives():
    qrels = {"q": {"a": 1, "b": 1, "z": 0}, "empty": {"a": 0}}
    res = recall_at_k({"q": {"a": 2, "c": 1}}, qrels, k=100)
    assert res == {"q": 0.5}
    assert mean(res) == 0.5


def test_paired_bootstrap_constant_difference():
    a = {str(i): 0.8 for i in range(50)}
    b = {str(i): 0.5 for i in range(50)}
    m, lo, hi = paired_bootstrap(a, b, n=200)
    assert abs(m - 0.3) < 1e-9 and abs(lo - 0.3) < 1e-9 and abs(hi - 0.3) < 1e-9
