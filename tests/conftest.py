from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DISASTER = [
    "Forest fire near La Ronge Sask. Canada",
    "13,000 people receive #wildfires evacuation orders in California",
    "Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school",
    "Earthquake of magnitude 6.2 hits the coast, buildings collapsed",
    "Flood warning issued as river bursts its banks, hundreds evacuated",
    "Suicide bomber kills 12 at a market in the capital",
    "Typhoon devastates coastal villages, thousands homeless",
    "Train derailment leaves 30 injured, emergency services on scene",
    "Massive explosion at chemical plant, residents told to stay indoors",
    "Wildfire spreads to 5000 hectares overnight, more evacuations ordered",
]
NOT_DISASTER = [
    "What a goooooaaaal!!!! that was on fire",
    "my phone battery just exploded in usage today lol",
    "this new album is a bomb, absolutely love it",
    "Crushed my workout this morning, feeling great",
    "The traffic today is an absolute disaster, late again",
    "I'm dying of laughter at this meme",
    "That exam was a bloodbath but we survived",
    "Summer heatwave? more like pool party season",
    "New job, new city, new me. Cannot wait",
    "Our team collapsed in the second half, typical",
]


@pytest.fixture
def small_df() -> pd.DataFrame:
    rows = [(i, "fire" if i % 2 == 0 else None, t, 1) for i, t in enumerate(DISASTER)]
    rows += [(100 + i, None, t, 0) for i, t in enumerate(NOT_DISASTER)]
    return pd.DataFrame(rows, columns=["id", "keyword", "text", "target"])


@pytest.fixture
def texts_and_labels(small_df):
    # Repeat to give the linear models something to fit on.
    df = pd.concat([small_df] * 4, ignore_index=True)
    return df["text"].tolist(), df["target"].to_numpy(), df


@pytest.fixture
def rng():
    return np.random.default_rng(0)
