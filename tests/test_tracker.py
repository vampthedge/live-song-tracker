import torch
import torch.nn.functional as F

from live_song_tracker import Match, Tracker
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
