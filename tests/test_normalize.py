from sentiment_eval.normalize import normalize_label


def test_exact_labels():
    assert normalize_label("positive") == "positive"
    assert normalize_label("NEUTRAL") == "neutral"
    assert normalize_label("negative!") == "negative"


def test_embedded_in_sentence():
    assert normalize_label("The sentiment is clearly positive.") == "positive"
    assert normalize_label("I would say this is negative overall") == "negative"


def test_unparseable():
    assert normalize_label("happy") is None
    assert normalize_label("") is None
    assert normalize_label(None) is None
