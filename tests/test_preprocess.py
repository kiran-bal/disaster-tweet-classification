from disaster_tweets.preprocess import dedupe_key, normalise


def test_urls_mentions_and_hashtags():
    out = normalise("Fire near @city! see http://t.co/abc #wildfire")
    assert out == "fire near user ! see url wildfire"


def test_html_entities_and_whitespace():
    assert normalise("Smoke &amp; ash   everywhere\n\n") == "smoke & ash everywhere"


def test_lowercase_can_be_disabled():
    assert normalise("Big Fire", lowercase=False) == "Big Fire"


def test_numbers_optional():
    assert "NUM" in normalise("12 dead", replace_numbers=True, lowercase=False)
    assert "12" in normalise("12 dead")


def test_none_and_empty_are_safe():
    assert normalise(None) == ""
    assert normalise("   ") == ""


def test_dedupe_key_ignores_trailing_urls_and_punctuation():
    a = dedupe_key("Forest fire near La Ronge! http://t.co/x")
    b = dedupe_key("forest fire near la ronge")
    assert a == b
