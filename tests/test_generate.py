from hinglish_retrieval.train.generate import _variants, drop_leaks, parse

GOOD = """HINDI: भारत का सबसे लंबा बांध कौन सा है?
HINGLISH 1: bharat ka sabse lamba dam kaunsa hai
HINGLISH 2: india ka sbse lamba baandh konsa h"""


def test_parse_good_answer():
    hi, hs = parse(GOOD)
    assert hi.startswith("भारत") and len(hs) == 2


def test_parse_markdown_and_thinking_wrappers():
    raw = "<think>\n</think>\n**HINDI:** भारत का सबसे लंबा बांध कौन सा है?\n**HINGLISH 1:** bharat ka sabse lamba dam kaunsa hai\n"
    hi, hs = parse(raw.replace("<think>\n</think>\n", ""))
    assert hi.startswith("भारत") and hs == ["bharat ka sabse lamba dam kaunsa hai"]


def test_parse_rejects_bad_answers():
    assert parse("I cannot help with that.") is None
    assert parse("HINDI: what is the longest dam\nHINGLISH 1: sabse lamba dam") is None  # not Devanagari
    assert parse("HINDI: भारत का सबसे लंबा बांध कौन सा है?\nHINGLISH 1: भारत ka dam") is None  # mixed script
    passage = "भारत का सबसे लंबा बांध कौन सा है? यह प्रश्न अक्सर पूछा जाता है।"
    assert parse(GOOD, passage) is None  # question copied from the passage


def test_variants_add_rule_based_spelling():
    out = _variants("भारत का सबसे लंबा बांध कौन सा है?", ["bharat ka sabse lamba dam kaunsa hai"], "q1", 1)
    assert len(out) == 2 and out[0] == "bharat ka sabse lamba dam kaunsa hai"


def test_drop_leaks_removes_eval_overlap():
    items = [
        {"text_native": "भारत का राष्ट्रीय पशु कौन सा है", "text_hinglish": ["x y z"]},
        {"text_native": "जापान की राजधानी क्या है", "text_hinglish": ["japan ki rajdhani kya hai"]},
        {"text_native": "सबसे ऊँचा पर्वत", "text_hinglish": ["bharat ka rashtriya pashu kaun sa hai bhai"]},
    ]
    keep, dropped = drop_leaks(items, {"e1": "bharat ka rashtriya pashu kaun sa hai"})
    assert dropped == 2 and keep == [items[1]]


def test_drop_leaks_exact_only_set():
    items = [{"text_native": "जापान की राजधानी क्या है", "text_hinglish": ["jabki yah jainon se km hai"]}]
    from hinglish_retrieval.leakage import normalise

    keep, dropped = drop_leaks(items, {}, {normalise("Jabki yah Jainon se km hai.")})
    assert dropped == 1 and keep == []
