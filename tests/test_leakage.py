from hinglish_retrieval.leakage import near_duplicates, normalise


def test_normalise_cross_script_and_repeats():
    assert normalise("किस यंत्र में?") == normalise("Kis   yantra men!!")
    assert normalise("kyaaa haal hai") == normalise("kya hal hai")


def test_near_duplicates_finds_planted_overlap():
    test = {"t1": normalise("bharat ka rashtriya pashu kaun sa hai"), "t2": normalise("cricket world cup")}
    train = {
        "a": normalise("bharat ka rashtriy pashu kaun sa hai"),  # planted near copy
        "b": normalise("japan ki rajdhani kya hai"),
    }
    hits = near_duplicates(train, test, threshold=0.7)  # one-letter edit scores ~0.77
    assert [(h[0], h[1]) for h in hits] == [("a", "t1")]
    assert near_duplicates(train, test, threshold=0.8) == []
