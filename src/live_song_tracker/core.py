"""Follow a live concert recording through a library of studio recordings: which song is playing, and where in it.

Every 3 s the last 10 s of live audio go through PupuM2D, a self-supervised music encoder.
Its output is cut into 0.48 s "sticks" and slid along every studio song. One window alone often lands
on the wrong chorus or verse repeat, so the tracker keeps an evidence score for every studio stick of
every song, moves it forward with the concert, lets it fade, and adds each new window's scores.

    live-song-tracker --library studio/ --live show.wav
"""

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from timm.layers import create_rope_embed
from timm.models.eva import EvaBlock
from torchcodec.decoders import AudioDecoder

WINDOW_SECONDS = 10  # the model was trained on 30 s inputs, so 10 s windows fit well inside
STEP_SECONDS = 3  # a new window every 3 s
WINDOWS_PER_FORWARD_PASS = 4

MODEL_NAME = "spellbrush/PupuM2D"
MODEL_REVISION = "c0f9345a1de0e61237dec7da32a70049c4ec0fa8"
MODEL_SUBFOLDER = "pupum2dV2_25hz_large"
MODEL_CKPT = "pupum2dV2_25hz_large/checkpoint/step-0500000_loss-0.176985/model.safetensors"
MODEL_SAMPLE_RATE = 24000
STICK_SECONDS = 0.48
FRAMES_PER_STICK = 12  # PupuM2D gives 25 frames per second, so 12 frames make one 0.48 s stick

SONG_SCORE_NEEDED = 0.7  # below this pooled score nothing in the library sounds like the live window
EVIDENCE_DECAY = 0.9  # how much of last step's evidence survives one step: 0 forgets everything, 1 never forgets
CONFIDENCE_NEEDED = 0.3  # below this the pick is usually a "magnet" stick that scores well against anything
JUMP_MARGIN = 0.1  # to jump more than 6 s away, the new pick's evidence must beat the expected spot's by this much
STICKS_PER_STEP = round(STEP_SECONDS / STICK_SECONDS)  # 3 s = 6 sticks

AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".aiff", ".ogg"]


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(path, sample_rate=MODEL_SAMPLE_RATE):
    """Decode any audio file into one mono 1D tensor at sample_rate."""
    decoder = AudioDecoder(str(path), sample_rate=sample_rate, num_channels=1)
    return decoder.get_all_samples().data[0]


def cut_into_windows(samples, sample_rate=MODEL_SAMPLE_RATE, window_seconds=WINDOW_SECONDS):
    """(number_of_windows, samples_per_window). The short tail is dropped; a clip shorter than one window is padded."""
    samples_per_window = window_seconds * sample_rate
    number_of_windows = len(samples) // samples_per_window
    if number_of_windows == 0:
        return F.pad(samples, (0, samples_per_window - len(samples))).view(1, samples_per_window)
    return samples[: number_of_windows * samples_per_window].view(number_of_windows, samples_per_window)


# PatchEmbedRoPE, PupuM2DEncoder and PupuM2DModel are adapted from https://github.com/sizigi/PupuM2D
# (MIT License, Copyright (c) 2026 PupuM2D contributors).


class PatchEmbedRoPE(nn.Module):
    def __init__(self, img_size=(1024, 128), patch_size=(4, 16), in_chans=1, embed_dim=1024):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        outputs = self.proj(x).permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        return self.norm_layer(outputs)


class PupuM2DEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, use_swiglu=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            EvaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                norm_layer=nn.LayerNorm, attn_type="rope", num_prefix_tokens=0,
                drop_path=0.0, swiglu_mlp=use_swiglu, init_values=None, qk_norm=False,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, rope):
        start_layer = len(self.blocks) // 2 - 1  # paper Table II: layers 12..24 (1-indexed) for PupuM2D-Large
        selected_layers = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x, rope=rope)
            if idx == len(self.blocks) - 1:
                x = self.norm(x)
            if idx >= start_layer:
                selected_layers.append(F.layer_norm(x, (x.shape[-1],), eps=1e-6))
        return torch.stack(selected_layers, dim=0).mean(dim=0)


class PupuM2DModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        self.patch_size = tuple(mcfg["patch_size"])
        self.embed_dim = mcfg["embed_dim"]
        self.patch_embed = PatchEmbedRoPE(
            img_size=tuple(mcfg["image_size"]),
            patch_size=self.patch_size,
            in_chans=mcfg["in_chans"],
            embed_dim=self.embed_dim,
        )
        self.rope_encoder = create_rope_embed(
            rope_type="cat", dim=self.embed_dim, num_heads=mcfg["num_heads"], feat_shape=None
        )
        self.teacher = PupuM2DEncoder(
            embed_dim=self.embed_dim,
            depth=mcfg["depth"],
            num_heads=mcfg["num_heads"],
            mlp_ratio=mcfg["mlp_ratio"],
            use_swiglu=mcfg["use_swiglu"],
        )
        pcfg = cfg["preprocess"]
        # slaney scale and norm: the same filters librosa's mel() gives, which the model was trained with
        mel = torchaudio.functional.melscale_fbanks(
            n_freqs=pcfg["n_fft"] // 2 + 1, f_min=pcfg["fmin"], f_max=pcfg["fmax"], n_mels=pcfg["n_mels"],
            sample_rate=pcfg["sample_rate"], norm="slaney", mel_scale="slaney",
        )  # (frequencies, mels)
        self.register_buffer("mel_basis", mel.T.float(), persistent=False)  # (mels, frequencies)
        self.register_buffer("hann_window", torch.hann_window(pcfg["win_size"]), persistent=False)

    def extract_mel(self, y):
        pcfg = self.cfg["preprocess"]
        pad = (pcfg["n_fft"] - pcfg["hop_size"]) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y, pcfg["n_fft"], hop_length=pcfg["hop_size"], win_length=pcfg["win_size"],
            window=self.hann_window, center=False, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        if pcfg["normalize"]:
            spec = (spec - (-4.089994845986366)) / (2.0242277159094813 + 1e-8)
        if pcfg["flip_ft"]:
            spec = spec.transpose(-2, -1)
        t_frames = (spec.shape[-2] // self.patch_size[0]) * self.patch_size[0]
        spec = spec[:, :t_frames, :]
        return spec.unsqueeze(1)

    def rope_embed(self, grid_size):
        """RoPE positions with the patch spacing the model was trained with.

        timm spreads however many time patches it gets over positions -1..1, and training always
        used image_size[0] mel frames (256 patches). Cropping that training grid keeps the training
        distance between neighbouring patches for shorter inputs.
        """
        train_rows = self.cfg["model"]["image_size"][0] // self.patch_size[0]
        rows, cols = grid_size
        if rows > train_rows:
            return self.rope_encoder.get_embed(grid_size)
        full = self.rope_encoder.get_embed((train_rows, cols))  # (train_rows * cols, D)
        return full.view(train_rows, cols, -1)[:rows].reshape(rows * cols, -1)

    def forward(self, input_values):
        """Raw audio (B, samples) -> (B, time_patches, frequency_patches * D), 25 time patches per second."""
        imgs = self.extract_mel(input_values)
        B, _, H, W = imgs.shape
        grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        x = self.patch_embed(imgs)
        rope = self.rope_embed(grid_size).to(x.device).unsqueeze(0)
        h = self.teacher(x, rope=rope)
        return h.view(B, grid_size[0], grid_size[1] * self.embed_dim)


def load_model(device=None):
    """Download (once) and load PupuM2D-Large. device: "cuda", "mps" or "cpu"; default: the fastest available."""
    args_path = hf_hub_download(repo_id=MODEL_NAME, filename=f"{MODEL_SUBFOLDER}/args.json", revision=MODEL_REVISION)
    ckpt_path = hf_hub_download(repo_id=MODEL_NAME, filename=MODEL_CKPT, revision=MODEL_REVISION)
    with open(args_path) as f:
        cfg = json.loads(re.sub(r",(\s*[\]}])", r"\1", f.read()))  # args.json has trailing commas
    model = PupuM2DModel(cfg)
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("patch_embed.") or k.startswith("teacher."):
                state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict, strict=True)
    return model.train(False).to(device or default_device())  # inference mode


def embed_windows(windows, model):
    """Windows (N, samples) -> pooled (N, C), one vector per window, and sticks (N, sticks_per_window, C). Both unit length."""
    device = next(model.parameters()).device
    pooled, sticks = [], []
    for start in range(0, len(windows), WINDOWS_PER_FORWARD_PASS):
        with torch.no_grad():
            hidden = model(windows[start : start + WINDOWS_PER_FORWARD_PASS].to(device))  # (B, T, C)
        B, T, C = hidden.shape
        sticks_per_window = T // FRAMES_PER_STICK  # 10 s = 250 frames -> 20 sticks, the last 10 frames are dropped
        frames_used = sticks_per_window * FRAMES_PER_STICK
        pooled.append(hidden.mean(dim=1).cpu())
        sticks.append(hidden[:, :frames_used].reshape(B, sticks_per_window, FRAMES_PER_STICK, C).mean(dim=2).cpu())
    return F.normalize(torch.cat(pooled), dim=-1), F.normalize(torch.cat(sticks), dim=-1)


def build_library(library_folder, model, cache=None):
    """Embed every studio recording in a folder; the file name is the song name.

    Returns (pooled, sticks): song -> (windows, C) and song -> (windows, sticks_per_window, C).
    With cache, a saved library is loaded from that path, or the new one is saved there.
    """
    if cache and Path(cache).exists():
        return torch.load(cache, weights_only=True)
    pooled, sticks = {}, {}
    for path in sorted(Path(library_folder).iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        pooled[path.stem], sticks[path.stem] = embed_windows(cut_into_windows(load_audio(path)), model)
        print(f"  {path.stem}: {len(pooled[path.stem])} windows")
    if not pooled:
        raise ValueError(f"No audio files found in {library_folder}")
    if cache:
        torch.save((pooled, sticks), cache)
    return pooled, sticks


def score_songs(live_pooled, library_pooled):
    """Per song: the best-matching studio window for every live window, averaged."""
    return {
        song: (live_pooled @ song_pooled.T).max(dim=1).values.mean().item()
        for song, song_pooled in library_pooled.items()
    }


def center_sticks(sticks, mean_stick):
    """Subtract the library's average stick and normalize again.

    Every stick of every song points roughly the same way ("loud rock music"), so raw cosine similarity
    is about 0.7 everywhere. Removing the shared part leaves what makes this moment different.
    """
    return F.normalize(sticks - mean_stick, dim=-1)


def alignment_scores(live_sticks, song_sticks):
    """Slide the live sticks along the studio song: one score per possible start stick.

    live_sticks (L, C), song_sticks (S, C) -> (S - L + 1,), the average similarity of live stick i with
    studio stick start + i.
    """
    similarity = live_sticks @ song_sticks.T  # (L, S)
    L, S = similarity.shape
    steps = torch.arange(L)
    starts = torch.arange(S - L + 1)
    return similarity[steps, starts[:, None] + steps].mean(dim=1)


def stick_to_second(stick_index):
    """Studio second where a stick starts. Each 10 s window holds 20 sticks of 0.48 s (9.6 s), the rest is dropped."""
    sticks_per_window = (WINDOW_SECONDS * 25) // FRAMES_PER_STICK
    return (stick_index // sticks_per_window) * WINDOW_SECONDS + (stick_index % sticks_per_window) * STICK_SECONDS


class Match(NamedTuple):
    song: str  # library file name without extension
    second: float  # position in the studio recording
    confidence: float  # average window similarity behind this pick
    sure: bool  # confidence >= CONFIDENCE_NEEDED; below it the pick is usually wrong


class Tracker:
    """Follows one live performance. Feed it a WINDOW_SECONDS window every STEP_SECONDS.

    tracker = Tracker(model, build_library("studio/", model))
    match = tracker.step(last_10_seconds)  # 1D tensor at MODEL_SAMPLE_RATE, mono
    """

    def __init__(self, model, library):
        self.model = model
        self.library_pooled, library_sticks = library
        C = next(iter(library_sticks.values())).shape[-1]
        self.mean_stick = torch.cat([sticks.view(-1, C) for sticks in library_sticks.values()]).mean(dim=0)
        self.centered_library = {song: center_sticks(sticks.view(-1, C), self.mean_stick) for song, sticks in library_sticks.items()}
        self.evidence = {}  # song -> one score per possible start stick
        self.evidence_weight = 0.0  # 1, 1.9, 2.71, ... settles at 1 / (1 - decay)
        self.last_song = ""
        self.last_stick = -1

    def step(self, window):
        """One live window, WINDOW_SECONDS of mono audio at MODEL_SAMPLE_RATE (shorter is padded with silence).

        Returns a Match, or None when no song in the library sounds like this window.
        """
        samples = WINDOW_SECONDS * MODEL_SAMPLE_RATE
        window = F.pad(window[:samples], (0, max(0, samples - len(window))))
        return self.step_embeddings(*embed_windows(window.view(1, -1), self.model))

    def step_embeddings(self, live_pooled, live_sticks):
        """step() after the model: live_pooled (1, C) and live_sticks (1, sticks, C) from embed_windows."""
        if max(score_songs(live_pooled, self.library_pooled).values()) <= SONG_SCORE_NEEDED:
            # shortcut: a skipped window does not move the evidence on (as in the original notebook), so the
            # next pick compares against evidence one step behind; shift it here too if skips mid-song get common
            return None
        live_sticks = center_sticks(live_sticks.view(-1, live_sticks.shape[-1]), self.mean_stick)
        for song, song_sticks in self.centered_library.items():
            new_scores = alignment_scores(live_sticks, song_sticks)
            moved = torch.zeros_like(new_scores)
            if song in self.evidence:
                moved[STICKS_PER_STEP:] = self.evidence[song][:-STICKS_PER_STEP]  # the concert moved on one step
            self.evidence[song] = EVIDENCE_DECAY * moved + new_scores

        song = max(self.evidence, key=lambda s: self.evidence[s].max())
        stick = int(self.evidence[song].argmax())
        # last step's pick, moved on by one step, is where we expect to be now. A pick elsewhere
        # (another song, or more than 6 s away) must beat the evidence there by JUMP_MARGIN
        if self.last_stick != -1:
            expected = min(self.last_stick + STICKS_PER_STEP, len(self.evidence[self.last_song]) - 1)
            is_far = song != self.last_song or abs(stick_to_second(stick) - stick_to_second(expected)) > 6
            if is_far and self.evidence[song][stick] < self.evidence[self.last_song][expected] * (1 + JUMP_MARGIN):
                song, stick = self.last_song, expected
        self.last_song, self.last_stick = song, stick
        # evidence is a weighted sum of window scores; dividing by the weights gives their average
        self.evidence_weight = EVIDENCE_DECAY * self.evidence_weight + 1
        confidence = self.evidence[song][stick].item() / self.evidence_weight
        return Match(song, stick_to_second(stick), confidence, confidence >= CONFIDENCE_NEEDED)


def load_lyrics(path):
    """lrclib.net JSON (its "syncedLyrics" LRC text) -> list of (second, text); [] when missing."""
    if not path.exists():
        return []
    lines = []
    for line in (json.loads(path.read_text())["syncedLyrics"] or "").split("\n"):
        if not line.startswith("["):
            continue
        time_part, text = line[1:].split("]", 1)  # "00:25.31" and " This must all be..."
        minutes, seconds = time_part.split(":")
        lines.append((int(minutes) * 60 + float(seconds), text.strip()))
    return lines


def lyric_at(lines, second):
    """The line being sung at `second` of the studio recording; "" in the intro or without lyrics."""
    current = ""
    for line_second, text in lines:
        if line_second <= second:
            current = text
    return current


def load_setlist(path):
    """Tab-separated song<TAB>start_sec<TAB>end_sec per line, # for comments -> {song: (start, end)}."""
    setlist = {}
    for line in path.read_text().split("\n"):
        if line.strip() and not line.startswith("#"):
            song, start, end = line.strip().split("\t")
            setlist[song] = (int(start), int(end))
    return setlist


def format_time(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--library", required=True, help="folder with one studio recording per song")
    parser.add_argument("--live", required=True, help="concert recording")
    parser.add_argument("--setlist", help="song<TAB>start<TAB>end per line; default: the live file with .txt")
    parser.add_argument("--lyrics", help="folder of lrclib.net JSON files named <song>.json")
    parser.add_argument("--cache", help="save the embedded library here and reuse it on later runs")
    parser.add_argument("--start", type=float, default=0, help="live second to start at")
    parser.add_argument("--steps", type=int, help="number of 3 s steps (default: to the end)")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} on {default_device()}")
    model = load_model()
    try:
        tracker = Tracker(model, build_library(args.library, model, cache=args.cache))
    except ValueError as error:
        raise SystemExit(error)

    live_path = Path(args.live)
    setlist_path = Path(args.setlist) if args.setlist else live_path.with_suffix(".txt")
    setlist = load_setlist(setlist_path) if setlist_path.exists() else {}
    lyrics = {}

    samples = load_audio(live_path)
    window = WINDOW_SECONDS * MODEL_SAMPLE_RATE
    starts = range(int(args.start * MODEL_SAMPLE_RATE), max(1, len(samples) - window + 1), STEP_SECONDS * MODEL_SAMPLE_RATE)
    for start_sample in list(starts)[: args.steps]:
        start_second = start_sample / MODEL_SAMPLE_RATE
        expected = next(
            (f"{song} at {format_time(start_second - s)}" for song, (s, e) in setlist.items()
             if s <= start_second and e >= start_second + WINDOW_SECONDS),
            "",
        )
        match = tracker.step(samples[start_sample : start_sample + window])
        line = f"live {format_time(start_second)}  "
        if match is None:
            print(line + "no song in the library sounds like this")
            continue
        song, second, confidence, sure = match
        line += f"{song} at {format_time(second)} ({second:.1f}s) confidence {confidence:.2f}"
        if not sure:
            line += "  (not sure)"
        elif args.lyrics:
            if song not in lyrics:
                lyrics[song] = load_lyrics(Path(args.lyrics) / f"{song}.json")
            line += f"  | {lyric_at(lyrics[song], second) or '♪'}"
        if expected:
            line += f"  [setlist: {expected}]"
        print(line)


if __name__ == "__main__":
    main()
