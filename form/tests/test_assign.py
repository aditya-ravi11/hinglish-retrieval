from assign import RESERVE_SECONDS, Assigner, Source


def make(n=4):
    return [Source(f"q{i}", "hi", "test", f"text {i}", priority=i) for i in range(n)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_never_repeats_for_same_annotator():
    a = Assigner(make(3))
    seen = []
    while (s := a.next_for("alice")) is not None:
        seen.append(s.source_id)
        a.record("alice", s.source_id, skipped=False)
    assert len(seen) == len(set(seen)) == 3


def test_completes_pairs_first():
    a = Assigner(make(4))
    s = a.next_for("alice")
    a.record("alice", s.source_id, skipped=False)
    # bob should get the query alice already did, to finish the pair
    assert a.next_for("bob").source_id == s.source_id


def test_two_distinct_annotators_per_query_then_cap_rises():
    a = Assigner(make(2))
    for who in ["a", "b", "c", "d"]:
        while (s := a.next_for(who)) is not None:
            a.record(who, s.source_id, skipped=False)
            if a.complete_pairs() == 2 and who in ("a", "b"):
                pass
    # after everything has 2, the cap rises to 3 and then stops
    for sid in ["q0", "q1"]:
        assert len(a.rewrites[sid]) == 3


def test_reservation_blocks_concurrent_overfill_and_expires():
    clock = Clock()
    a = Assigner(make(1), clock=clock)
    a.record("x", "q0", skipped=False)  # q0 has 1 rewrite
    assert a.next_for("alice").source_id == "q0"  # alice reserves the 2nd slot
    assert a.next_for("bob") is None  # q0 is full (1 done + 1 reserved)
    clock.t += RESERVE_SECONDS + 1
    assert a.next_for("bob").source_id == "q0"  # alice's reservation expired


def test_reload_returns_same_query():
    a = Assigner(make(3))
    first = a.next_for("alice")
    assert a.next_for("alice").source_id == first.source_id


def test_retire_after_three_skips():
    a = Assigner(make(1))
    for who in ["a", "b", "c"]:
        s = a.next_for(who)
        a.record(who, s.source_id, skipped=True)
    assert a.next_for("d") is None
