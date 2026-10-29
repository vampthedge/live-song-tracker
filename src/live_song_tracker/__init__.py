"""Follow a live concert recording through a library of studio recordings: which song is playing, and where in it."""

from .core import (
    MODEL_SAMPLE_RATE,
    STEP_SECONDS,
    WINDOW_SECONDS,
    Match,
    Tracker,
    build_library,
    load_audio,
    load_lyrics,
    load_model,
    lyric_at,
)

__all__ = [
    "MODEL_SAMPLE_RATE",
    "STEP_SECONDS",
    "WINDOW_SECONDS",
    "Match",
    "Tracker",
    "build_library",
    "load_audio",
    "load_lyrics",
    "load_model",
    "lyric_at",
]
