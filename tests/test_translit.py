from hinglish_retrieval.data.translit import machine_romanize


def test_examples():
    assert machine_romanize("मुझे सस्ता फ़ोन चाहिए") == "mujhe sasta fon chahie"
    assert machine_romanize("किस यंत्र में?") == "kis yantra men"
    assert machine_romanize("भारत के राष्ट्रीय चिन्ह") == "bharat ke rashtriya chinh"


def test_output_is_plain_ascii():
    out = machine_romanize("अल्जीरिया किस मरुस्थल के अंतर्गत स्थित है?")
    assert out.isascii() and out == out.lower()
