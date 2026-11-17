import random

from hinglish_retrieval.train.noise import EN_SWAP, hinglish_noisy

Q = "भारत का राष्ट्रीय पशु कौन सा है?"


def test_same_seed_same_spelling():
    assert hinglish_noisy(Q, random.Random(3)) == hinglish_noisy(Q, random.Random(3))


def test_output_is_latin_lowercase():
    for s in range(20):
        out = hinglish_noisy(Q, random.Random(s))
        assert out and out.isascii() and out == out.lower()


def test_seeds_give_different_spellings():
    assert len({hinglish_noisy(Q, random.Random(s)) for s in range(20)}) > 3


def test_english_swap_keeps_english_spelling():
    out = hinglish_noisy("प्रसिद्ध नदी कौन सी है", random.Random(0), p_en=1.0)
    assert out.split()[:2] == [EN_SWAP["प्रसिद्ध"], EN_SWAP["नदी"]]


def test_no_noise_is_close_to_machine_romanization():
    out = hinglish_noisy("ज्ञान का अर्थ", random.Random(0), p=0.0, p_en=0.0)
    assert out.split()[1] == "ka"
