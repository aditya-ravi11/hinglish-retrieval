from hinglish_retrieval.data.backtranslit import clean


def test_clean_keeps_first_devanagari_line():
    assert clean("Query: भारत का राष्ट्रीय पशु\nThis means...", "bharat ka rashtriya pashu") == "भारत का राष्ट्रीय पशु"


def test_clean_falls_back_to_typed_query():
    assert clean("Sorry, I can't.", "tum kaisa ho") == "tum kaisa ho"


def test_clean_cuts_loops():
    out = clean("गुलाम " + "जान " * 30, "gulam ki kabr")
    assert len(out.split()) <= 3 + 3
