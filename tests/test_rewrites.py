from hinglish_retrieval.data.rewrites import writer_slots


def row(anon, src, text, t, skipped=False):
    return {"type": "rewrite", "anon_id": anon, "source_id": src, "rewrite_raw": text,
            "skipped": skipped, "submitted_at": t}


def test_two_distinct_writers_oldest_first():
    rows = [
        row("b", "q", "kya hal hai", "2026-11-12T10:00:02"),
        row("a", "q", "kia haal h", "2026-11-12T10:00:01"),
        row("a", "q", "duplicate by a", "2026-11-12T10:00:03"),  # same writer again: ignored
        row("c", "q", "third writer", "2026-11-12T10:00:04"),  # slots already full
        row("d", "q", None, "2026-11-12T10:00:00", skipped=True),  # skips never count
        {"type": "consent", "anon_id": "a"},
    ]
    slots = writer_slots(rows)
    assert [r["rewrite_raw"] for r in slots["q"]] == ["kia haal h", "kya hal hai"]


def test_rewrite_text_prefers_cleaned():
    from hinglish_retrieval.data.rewrites import rewrite_text

    assert rewrite_text({"rewrite_raw": "[synthetic] kya", "rewrite": "kya"}) == "kya"
    assert rewrite_text({"rewrite_raw": "kya"}) == "kya"
