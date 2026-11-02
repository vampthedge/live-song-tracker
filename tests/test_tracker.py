import pytest
import torch
import torch.nn.functional as F

from live_song_tracker import Match, Tracker, core
from live_song_tracker.core import STICKS_PER_STEP, alignment_scores, stick_to_second


def test_alignment_scores_matches_the_diagonal_loop():
    live, song = torch.randn(20, 8), torch.randn(50, 8)
    similarity = live @ song.T
    loop = [sum(similarity[i, start + i] for i in range(20)) / 20 for start in range(50 - 20 + 1)]
    assert torch.allclose(alignment_scores(live, song), torch.stack(loop), atol=1e-6)


def synthetic_library():
    torch.manual_seed(0)
    pooled = {name: F.normalize(torch.randn(20, 64), dim=-1) for name in ["a", "b", "c"]}
    sticks = {name: F.normalize(torch.randn(20, 20, 64), dim=-1) for name in ["a", "b", "c"]}  # 20 windows of 20 sticks
    return pooled, sticks


def test_tracker_follows_a_live_clip_cut_from_one_song():
    pooled, sticks = synthetic_library()
    tracker = Tracker(None, (pooled, sticks))
    song_b = sticks["b"].view(-1, 64)
    for step in range(5):
        start = 100 + step * STICKS_PER_STEP
        live = F.normalize(song_b[start : start + 20] + 0.06 * torch.randn(20, 64), dim=-1)
        match = tracker.step_embeddings(pooled["b"][5:6], live.view(1, 20, 64))
    assert match == Match("b", stick_to_second(start), match.confidence, True)
    assert match.confidence > 0.5


def test_tracker_skips_a_window_no_song_sounds_like():
    pooled, sticks = synthetic_library()
    tracker = Tracker(None, (pooled, sticks))
    unrelated = F.normalize(torch.randn(1, 64), dim=-1)
    assert tracker.step_embeddings(unrelated, sticks["a"][:1]) is None


def test_the_background_correction_stops_a_hub_song_winning(monkeypatch):
    """A hub song made of what every live window shares (crowd, PA, room) beats the faint true song without it."""

    def picks(weight):
        monkeypatch.setattr(core, "HUB_WEIGHT", weight)
        pooled, sticks = synthetic_library()
        room = F.normalize(torch.randn(64), dim=0)
        sticks["h"] = F.normalize(room + 0.3 * torch.randn(20, 20, 64), dim=-1)
        pooled["h"] = F.normalize(room + 0.3 * torch.randn(20, 64), dim=-1)
        tracker = Tracker(None, (pooled, sticks))
        song_b = sticks["b"].view(-1, 64)
        songs = []
        for step in range(20):
            start = 40 + step * STICKS_PER_STEP
            live = F.normalize(0.2 * song_b[start : start + 20] + room + 0.3 * torch.randn(20, 64), dim=-1)
            songs.append(tracker.step_embeddings(pooled["b"][5:6], live.view(1, 20, 64)).song)
        return songs

    assert set(picks(0.0)) == {"h"}
    assert set(picks(1.0)) == {"b"}


@pytest.mark.parametrize("weight", [1.0, 0.0])
def test_the_correction_keeps_following_one_song_and_its_confidence_scale(monkeypatch, weight):
    monkeypatch.setattr(core, "HUB_WEIGHT", weight)
    pooled, sticks = synthetic_library()
    tracker = Tracker(None, (pooled, sticks))
    song_b = sticks["b"].view(-1, 64)
    for step in range(5):
        start = 100 + step * STICKS_PER_STEP
        live = F.normalize(song_b[start : start + 20] + 0.06 * torch.randn(20, 64), dim=-1)
        match = tracker.step_embeddings(pooled["b"][5:6], live.view(1, 20, 64))
    assert (match.song, match.second, match.sure) == ("b", stick_to_second(start), True)
    assert match.confidence > 0.5  # the offset is added back: confidence stays plain similarity
