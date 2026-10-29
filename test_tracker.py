import torch
import torch.nn.functional as F

from live_song_tracker import STICKS_PER_STEP, Tracker, alignment_scores, stick_to_second


def test_alignment_scores_matches_the_diagonal_loop():
    live, song = torch.randn(20, 8), torch.randn(50, 8)
    similarity = live @ song.T
    loop = [sum(similarity[i, start + i] for i in range(20)) / 20 for start in range(50 - 20 + 1)]
    assert torch.allclose(alignment_scores(live, song), torch.stack(loop), atol=1e-6)


def test_tracker_follows_a_live_clip_cut_from_one_song():
    torch.manual_seed(0)
    library = {name: F.normalize(torch.randn(400, 64), dim=-1) for name in ["a", "b", "c"]}
    tracker = Tracker(library)
    for step in range(5):
        start = 100 + step * STICKS_PER_STEP
        live = F.normalize(library["b"][start : start + 20] + 0.5 * torch.randn(20, 64) / 8, dim=-1)
        song, second, confidence = tracker.step(live)
    assert (song, second) == ("b", stick_to_second(start))
    assert confidence > 0.5
