"""Text normalisation.

Deliberately light. Tweets are short, so aggressive cleaning (stemming, stop-word
removal) throws away signal that n-gram and transformer models can use. Every
step here is reversible in the sense that it only collapses tokens that carry
no class information on their own: URLs, user mentions, HTML entities, and
repeated whitespace.
"""

from __future__ import annotations

import html
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
# Non-printable / control characters that appear in the raw Kaggle export.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?\b")


def normalise(
    text: str,
    *,
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    strip_hashtag_symbol: bool = True,
    replace_numbers: bool = False,
) -> str:
    """Return a cleaned version of ``text``.

    The replacements use readable placeholder tokens (``URL``, ``USER``,
    ``NUM``) rather than deleting spans, so a model can still learn that
    "tweets with a link" behave differently from tweets without one.
    """
    if text is None:
        return ""
    out = html.unescape(str(text))
    out = unicodedata.normalize("NFKC", out)
    out = CONTROL_RE.sub(" ", out)
    if replace_urls:
        out = URL_RE.sub(" URL ", out)
    if replace_mentions:
        out = MENTION_RE.sub(" USER ", out)
    if strip_hashtag_symbol:
        out = HASHTAG_RE.sub(r"\1", out)
    if replace_numbers:
        out = NUMBER_RE.sub(" NUM ", out)
    if lowercase:
        out = out.lower()
    return WHITESPACE_RE.sub(" ", out).strip()


def dedupe_key(text: str) -> str:
    """Key used to detect duplicate tweets.

    Two tweets are the same if they match after normalisation and removal of
    every non-alphanumeric character. This catches retweets that differ only
    in a trailing URL or a stray punctuation mark.
    """
    base = normalise(text, replace_urls=False, replace_mentions=False)
    base = URL_RE.sub(" ", base)
    base = MENTION_RE.sub(" ", base)
    return re.sub(r"[^a-z0-9]+", "", base)
