from hinglish_retrieval.data.dakshina import order, sentence_keys
from hinglish_retrieval.leakage import normalise


def test_split_order_is_fixed_and_content_based():
    a = ["पहला वाक्य यहाँ है।", "दूसरा वाक्य यहाँ है।", "तीसरा वाक्य यहाँ है।"]
    perm = order(a)
    assert sorted(perm) == [0, 1, 2]
    # same sentences in another input order land in the same final order
    b = [a[2], a[0], a[1]]
    assert [b[i] for i in order(b)] == [a[i] for i in perm]


def test_sentence_keys_find_a_sentence_inside_a_passage():
    sent = "वर्ष 2000 में वेंकटरामन ने प्रयोगशाला में राइबोसोम की खोज की।"
    passage = "कुछ और बातें यहाँ लिखी हैं जो लंबी हैं। " + sent + " आगे की कहानी भी लंबी है।"
    assert sentence_keys(sent) <= sentence_keys(passage)
    assert normalise(sent.rstrip("।")) in sentence_keys(passage)
