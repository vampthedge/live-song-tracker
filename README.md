# live-song-tracker

Point it at a band's studio recordings and a recording of one of their concerts. Every 3 seconds it
says which song is playing and where in the studio version the band is, so you can show the lyric
being sung right now.

```
live 5:30  disposable_teens at 2:39 (159.1s) confidence 0.37  [setlist: disposable_teens at 0:10]
live 5:33  disposable_teens at 2:42 (162.4s) confidence 0.32  [setlist: disposable_teens at 0:13]
live 5:36  disposable_teens at 0:11 (11.4s) confidence 0.35  [setlist: disposable_teens at 0:16]
live 5:39  disposable_teens at 0:14 (14.3s) confidence 0.35  [setlist: disposable_teens at 0:19]
```

A visual walkthrough with figures from a real concert: https://vampthedge.github.io/live-song-tracker/
(source in [docs/index.html](docs/index.html)).

## How it works

1. **Embed.** [PupuM2D-Large](https://huggingface.co/spellbrush/PupuM2D), a self-supervised music
   encoder (MIT), turns 10 s of audio into 25 vectors per second. Those are averaged into 0.48 s
   "sticks". Every studio song is embedded once up front.
2. **Center.** Every stick of every rock song points roughly the same way, so raw cosine similarity
   is about 0.7 everywhere. Subtracting the library's average stick leaves what makes a moment
   different. On one show this moved the true position from best match 44% of the time to 57%.
3. **Align.** The 20 live sticks slide along each studio song; each start position gets the average
   similarity along that diagonal.
4. **Track.** One window alone often lands on the wrong repeat of a chorus. The tracker keeps an
   evidence score for every studio stick of every song. Each step it shifts the evidence forward by
   3 s, multiplies it by 0.9 and adds the new window's scores. The right position scores well at
   the same shifted spot step after step; a wrong repeat does not.
5. **Gate.** A jump to another song, or more than 6 s away, has to beat the expected position by
   10%. Evidence divided by its total weight is the confidence; below 0.3 the pick is usually a
   "magnet", a studio stick that matches anything, and no lyric is shown.

## Results

Offline, share of 3 s steps within 3 s of the true studio position (the setlist start plus a fitted
offset), 10 s windows:

| show | one window alone | tracker | jumps > 6 s, one window → tracker |
|---|---|---|---|
| Marilyn Manson, Antwerp 5:00-11:00 | 40 / 110 | 88 / 110 | 49 → 4 |
| Marilyn Manson, Newcastle 15:00-25:00 | 68 / 194 | 168 / 194 | 93 → 2 |
| Bad Omens 2:30-22:30 | 239 / 380 | 286 / 380 | 108 → 9 |
| FIR 2:30-22:30 | 219 / 385 | 313 / 385 | 146 → 11 |
| Ice Nine Kills 2:30-22:30 | 164 / 291 | 246 / 291 | 100 → 4 |

Lock-in takes 2 to 6 steps after a song starts, up to 13 when the live intro differs from the
studio one. The remaining misses are those first steps and live outros that run past the end of
the studio song. Fitted tempo came out at 1.00 for every song, so live tempo is not the problem.

Tried and dropped: searching only near the last position (it parks on magnets for good), stretching
the studio side 0.9 to 1.1 for tempo, penalizing magnet sticks by their average similarity to the
library, and resetting evidence when the pooled best song disagrees.

## Install

```bash
pip install git+https://github.com/vampthedge/live-song-tracker
# or, in a uv project
uv add git+https://github.com/vampthedge/live-song-tracker
```

Python 3.11+. The first run downloads the PupuM2D checkpoint from Hugging Face (3.4 GB). CUDA,
Apple MPS and CPU all work. No audio, setlists or lyrics ship with this package; bring your own.

## Command line

```bash
live-song-tracker --library studio/ --live show.wav --cache studio.pt
```

Embedding the studio songs takes a while; `--cache` keeps the embeddings for later runs.

| option | |
|---|---|
| `--library` | folder with one studio recording per song; the file name is the song name |
| `--live` | the concert recording |
| `--setlist` | `song<TAB>start_second<TAB>end_second` per line, to print where you expected to be. Defaults to the live file with `.txt` |
| `--lyrics` | folder of `<song>.json` files from the [lrclib.net](https://lrclib.net) API (`/api/get`), to print the line being sung |
| `--start`, `--steps` | live second to start at, and how many 3 s steps to run |

## Python

```python
from live_song_tracker import MODEL_SAMPLE_RATE, STEP_SECONDS, WINDOW_SECONDS, Tracker, build_library, load_audio, load_model

model = load_model()  # or load_model("cpu")
library = build_library("studio/", model, cache="studio.pt")
tracker = Tracker(model, library)  # one Tracker per live performance

live = load_audio("show.wav")  # mono float tensor at MODEL_SAMPLE_RATE (24 kHz)
window, step = WINDOW_SECONDS * MODEL_SAMPLE_RATE, STEP_SECONDS * MODEL_SAMPLE_RATE
for start in range(0, len(live) - window, step):
    match = tracker.step(live[start : start + window])
    if match and match.sure:
        print(match.song, match.second, match.confidence)
```

`Tracker.step` expects the last 10 s of audio, once every 3 s, in order. For a microphone or stream,
keep a 10 s buffer of 24 kHz mono samples and call `step` every 3 s. It returns `None` when nothing
in the library sounds like the window, otherwise a `Match(song, second, confidence, sure)`. Show a
position or lyric only when `sure` is true. `load_lyrics(path)` and `lyric_at(lines, second)` turn an
lrclib.net JSON file into the line sung at that second.

## Add it to your project with an AI coding assistant

Paste this into Claude Code, Cursor, Codex or similar, inside your project:

```text
Add live song recognition to this project with the live-song-tracker library
(https://github.com/vampthedge/live-song-tracker, MIT license). Read its README first.

1. Install it as a dependency with this project's package manager, e.g.
   `uv add git+https://github.com/vampthedge/live-song-tracker` or
   `pip install git+https://github.com/vampthedge/live-song-tracker`. It needs Python 3.11+.
2. Import it; do not copy its source into this project. The API:
   - load_model(device=None) -> model. Downloads a 3.4 GB checkpoint on first use.
   - build_library(folder, model, cache="library.pt") -> library. The folder holds one studio
     recording per song; the file name is the song name.
   - Tracker(model, library).step(window) -> Match(song, second, confidence, sure) or None.
     window is the last 10 s of mono audio at 24 kHz as a 1D torch tensor; call it every 3 s.
   - load_audio(path) decodes any audio file to that format.
   - load_lyrics(path) and lyric_at(lines, second) read lrclib.net synced-lyrics JSON.
3. Feed the tracker from wherever this project gets audio (a file, a microphone, a stream):
   keep a 10 s buffer, call step every 3 s, and show a song position or lyric only when
   match.sure is true.
4. Ask me where my studio recordings are. Do not download music from YouTube or other sites.
5. Add a test that cuts a 10 s clip from one of my studio recordings, feeds it to a Tracker
   three times 3 s apart, and checks the song name. Run it.
6. Commit the change on a new git branch and show me the diff summary.
```

## Tests

```bash
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE). The model classes are adapted from
[sizigi/PupuM2D](https://github.com/sizigi/PupuM2D) (MIT, see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)); please cite their paper,
[Frequency-Aware Self-Supervised Music Representation Learning](https://arxiv.org/abs/2606.25713),
if you use this.
