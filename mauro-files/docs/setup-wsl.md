# AI Toolkit on WSL2 (local setup)

How this fork runs locally on Mauro's PC: Windows 10 + WSL2 (Ubuntu 24.04), with
everything installed in the repo's `.venv/` and no container. The RunPod case uses
Ostris' official RunPod template and isn't covered here.

## Hardware / driver baseline

| Item | Value (2026-10-05) |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB VRAM, compute capability 8.9 (Ada) |
| Windows driver | 616.92 (CUDA UMD 13.4) |
| RAM | 64 GB (`.wslconfig`: `memory=64GB`, `swap=8GB`, `localhostForwarding=true`) |
| Distro | Ubuntu 24.04, Python 3.12, Node 24 (nvm), uv |
| torch in `.venv` | 2.13.0 + cu130 (chosen by `manager/spec.py`) |

The torch cu130 wheels need a driver that supports **CUDA ≥ 13.0**. Never install a
Linux NVIDIA driver inside WSL. The GPU is exposed by the Windows driver through
`/usr/lib/wsl/lib/libcuda.so.1`.

**After every Windows driver update, run `mauro-files/check.sh`.**

## Scripts (`mauro-files/`)

| Script | What it does |
|---|---|
| `install.sh [--force]` | Pre-flight checks (nvidia-smi, WSL libcuda, python, node, uv). It runs `python3 -m manager install` when `.venv/` is missing and `python3 -m manager sync` otherwise, then runs `check.sh`. |
| `run.sh` | `manager sync` (a no-op unless requirements changed), Triton pre-warm, then `manager launch --no-browser` (builds the UI and serves it on port 8675). |
| `check.sh` | Doctor table: WSL kernel, driver/CUDA version, GPU, libcuda, Node, UI port, `.venv`, torch build, real fp16/bf16 matmuls, a pinned-memory copy, VRAM, and the optional accelerators (triton, flash-attn, natten, torchcodec, bitsandbytes). |
| `scripts/common.sh` | Shared helpers: colors, `REPO_ROOT`, WSL detection, `TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib`. |

The scripts wrap the repo's built-in manager (`manager/`, see `manager/README.md`).
The manager owns the torch pins, the accelerator wheels and the UI dependency
install, so the scripts duplicate none of it. They **never call
`manager update`**, because it runs `git pull`. Syncing with upstream is done by
hand (see below).

## First install

```bash
cd ~/github/ai-toolkit
mauro-files/install.sh
```

What ends up where (all gitignored):

- `.venv/`: Python env (created by uv). The UI's job runner finds it through `ui/cron/pythonPath.ts`.
- `.node/`, `.ffmpeg/`, `.uv/`: portable copies the manager downloads only when needed. The system Node 24 and uv are used when present.
- `ui/node_modules`, `ui/.next`, `ui/dist`: UI build.
- `aitk_db.db`: UI SQLite database (jobs, settings).
- `datasets/`, `output/`: default dataset and training output folders. Both can be changed in the UI settings.

## Running the UI

```bash
mauro-files/run.sh
```

Then open **http://localhost:8675** in a browser on Windows. WSL2 forwards
localhost to Windows. The UI is not exposed to the LAN and has no auth token,
which is fine for local use. To expose it later, set `AI_TOOLKIT_AUTH` and add a
Windows `netsh interface portproxy` and firewall rule.

The first launch builds the Next.js app, so it takes a minute or two. Stop it with
Ctrl+C. Training jobs run as separate processes, so stopping the UI doesn't kill a
running job.

In the UI **Settings**, set your **Hugging Face token**. Gated models (FLUX.1-dev
and others) need it, and you must also accept each model's license on HF. Models
are cached in `~/.cache/huggingface`.

## Training on 16 GB VRAM

Most example configs (`config/examples/*_24gb.yaml`) target 24 GB. On 16 GB:

- Turn on **Low VRAM** (`model.low_vram: true`) and quantization of the
  transformer and text encoder (`quantize: true`, `quantize_te: true`). The UI
  shows these as toggles on the job page.
- Use **Cache Text Embeddings** / **Unload TE** where the model supports them.
- Keep `gradient_checkpointing: true`, batch size 1, and moderate LoRA rank (16–32).
- **Layer offloading** (`layer_offloading`) moves part of the model to system RAM.
  64 GB RAM makes this practical for the larger models, at a speed cost.
- Realistic picks: SD 1.5 / SDXL, Z-Image Turbo, FLUX.2-klein 4B, Wan 2.1 1.3B,
  Chroma / FLUX.1 with low_vram + quantize. 14B video models and Qwen-Image are
  heavy: use offloading or RunPod.

System RAM matters too: model loading and quantization run in RAM. Keep
`memory=64GB` in `%UserProfile%\.wslconfig`.

## Troubleshooting

- **`check.sh` fails on driver CUDA < 13.0**: update the Windows NVIDIA driver,
  or delete `.venv` and reinstall so the manager picks the cu126 fallback.
- **Pinned memory on WSL2**: `cudaHostRegister` is unreliable on WSL2 (ComfyUI
  needs `--disable-pinned-memory` for this). Here the dataloader's `pin_memory` is
  **off by default** (opt-in via `pin_memory: true` on the dataset config), so leave
  it off. `check.sh` runs a pinned-memory copy test. If that row shows WARN, never
  enable it.
- **Very slow training instead of OOM**: the Windows driver can spill VRAM into
  shared system memory. Lower the VRAM settings above. You can also set
  *CUDA - Sysmem Fallback Policy* → *Prefer No Sysmem Fallback* in the NVIDIA
  Control Panel to get a clean OOM instead. That setting is global and also
  affects ComfyUI.
- **`~/.bashrc` exports CUDA 12.8 in `LD_LIBRARY_PATH`** (needed by ComfyUI's
  SageAttention build). It doesn't conflict with torch cu130, which loads
  `*.so.13` libraries from its own `nvidia-*` wheels. `check.sh`'s matmul test
  confirms this.
- **Triton errors about libcuda**: `scripts/common.sh` exports
  `TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib`, and `run.sh` pre-warms Triton so a
  failure shows up at startup.
- **Port 8675 already in use**: another UI instance is running. Check with
  `ss -ltnp | grep 8675`.
- **Broken env**: `rm -rf .venv && mauro-files/install.sh`. The manager's state
  lives inside `.venv`, so this is a full reset.
- Diagnostics from the manager itself: `python3 -m manager doctor` and
  `python3 -m manager detect --json`.

## Syncing with upstream (ostris/ai-toolkit)

```bash
git fetch upstream
git merge upstream/main
mauro-files/install.sh          # re-syncs deps if requirements/torch pins changed
mauro-files/check.sh
```

`upstream` = `https://github.com/ostris/ai-toolkit.git`, `origin` = `mauroao/ai-toolkit`.
