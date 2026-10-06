# CLAUDE.md

This is Mauro's fork of AI Toolkit (`origin` = `mauroao/ai-toolkit`), tracking `upstream` = `ostris/ai-toolkit`. It's used for local LoRA training on his WSL2 PC. For RunPod he uses Ostris' official template, so there is no custom Docker image here.

To re-check what's custom vs. stock at any time:
```
git diff $(git merge-base HEAD upstream/main) HEAD --stat
```

## What's custom vs. stock

Everything is stock upstream AI Toolkit **except**:

- **`mauro-files/`**: local setup scripts and docs (see below).
- **`CLAUDE.md`**: this file.

Everything else (`toolkit/`, `extensions_built_in/`, `jobs/`, `manager/`, `ui/`, `config/`, `run.py`, `run_*.sh`, `README.md`, etc.) is unmodified upstream. Keep it that way: put local needs into `mauro-files/` instead of patching core code, so `git merge upstream/main` stays conflict-free.

## Local environment (WSL2)

- Windows 10 + WSL2 Ubuntu 24.04, **RTX 4060 Ti 16 GB** (Ada, sm_89), 64 GB RAM. The Windows driver was 616.92 / CUDA 13.4 as of 2026-10-05.
- Everything lives in **`.venv/`**, created by the repo's own manager (`python3 -m manager install`). It installs torch 2.13.0+cu130, flash-attn, NATTEN, torchcodec and the UI deps. The torch pins live in `manager/spec.py`. Don't change them locally.
- `mauro-files/install.sh`: first install / dependency re-sync (wraps `manager install|sync`), then runs `check.sh`.
- `mauro-files/run.sh`: `manager sync` + Triton pre-warm + `manager launch --no-browser`. The UI is at `http://localhost:8675`, reachable from the Windows browser only (no LAN exposure, no `AI_TOOLKIT_AUTH`).
- `mauro-files/check.sh`: doctor table (driver/CUDA ≥ 13.0, libcuda in `/usr/lib/wsl/lib`, torch build, real GPU matmuls, pinned-memory test, accelerators). Run it after every Windows driver update.
- `mauro-files/scripts/common.sh`: shared helpers (colors, `REPO_ROOT`, WSL detection, `TRITON_LIBCUDA_PATH`).
- The scripts **never call `manager update`** because it does a `git pull`. Upstream sync is manual: `git fetch upstream && git merge upstream/main && mauro-files/install.sh`.
- Full docs: `mauro-files/docs/setup-wsl.md` (16 GB VRAM training tips, WSL2 troubleshooting).

Related: the sibling repo `../ComfyUI` has the same `mauro-files/` layout and its own `.venv` (torch cu128 + SageAttention, built with the system CUDA 12.8 toolkit). The two envs are independent.

## WSL2 gotchas

- Pinned memory (`cudaHostRegister`) is unreliable on WSL2. The dataset `pin_memory` option is off by default upstream; leave it off.
- `~/.bashrc` puts CUDA 12.8 on `LD_LIBRARY_PATH` for ComfyUI. This doesn't conflict with the cu130 torch here.

## Guardrails

- Never run `git push`. Commit locally only, and only when asked.
- English for code, names and Markdown. Avoid code comments except for unusual situations.
- Don't modify upstream files to fix local issues if a `mauro-files/` script or a config setting can do it.
