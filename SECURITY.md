# Security policy

## Supported versions

Security fixes go to the latest release on `main`. Older versions do not get backports.

| Version | Supported |
|---|---|
| 0.1.x (`main`) | yes |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Security → Report a vulnerability](https://github.com/vampthedge/live-song-tracker/security/advisories/new).
Do not open a public issue, pull request or discussion for a security problem.

Include what you can of:

- the affected version or commit;
- the input that triggers it (an audio file, cache file, lyrics or setlist file, or command line);
- what happens and what you expected;
- a minimal proof of concept, if you have one.

You can expect an acknowledgement within 7 days and an assessment within 14 days. If the report is
confirmed, a fix and a GitHub security advisory follow, with credit to you unless you prefer to stay
anonymous.

## Scope

This project runs locally on files you give it. In scope:

- **Library cache files** (`--cache`, `build_library(cache=...)`). They are read with
  `torch.load(weights_only=True)`. A crafted cache file that runs code or escapes that restriction
  is in scope. Only load cache files you created yourself.
- **Lyrics JSON and setlist files.** Parsing them must not do anything beyond reading text.
- **The model download.** The checkpoint comes from Hugging Face at a pinned revision
  (`spellbrush/PupuM2D`, `c0f9345a`). A way to make the loader fetch or run something else is in scope.

Out of scope here, please report upstream:

- Audio decoding bugs: audio files are decoded by [TorchCodec](https://github.com/pytorch/torchcodec)
  and FFmpeg. Decoding untrusted audio carries FFmpeg's risks.
- Bugs in PyTorch, timm, safetensors or huggingface-hub themselves.
- The PupuM2D model and checkpoints: [sizigi/PupuM2D](https://github.com/sizigi/PupuM2D).

Wrong song or position matches are accuracy issues, not vulnerabilities; open a normal issue for those.

## Dependencies

Dependabot checks the locked dependencies in `uv.lock` weekly and opens pull requests for security
updates.
