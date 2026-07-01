<img width="2304" height="1848" alt="termite-circus" src="https://github.com/user-attachments/assets/f1feba1e-a4bf-4a37-a17d-b2c57389b10f" />

# termite-circus

Benchmark suite for the termite ML inference engine bundled in Antfly. Two
variants are compared against [llama.cpp](https://github.com/ggml-org/llama.cpp):

- **stable** — the `antfly` binary installed via brew (`brew install antflydb/taps/antfly`)
- **latest** — `antfly` built from the [antfly](https://github.com/antflydb/antfly) main repo's `zig/` tree with the latest zig

Both binaries expose a termite-zig-compatible `generate` subcommand (either as
`antfly inference generate ...` on latest or `antfly termite generate ...` /
bare `generate ...` on older builds). The bench harness probes for whichever
shape the binary actually supports.

## Quick start

```bash
brew install antflydb/taps/antfly       # stable variant
git clone https://github.com/antflydb/antfly antfly   # latest variant source
make download-gemma4-e4b                # ~5 GB GGUF, skips if already present
make download-gemma4-e4b-qat            # ~5 GB QAT GGUF, skips if already present
make bench                              # build + benchmark
```

`make bench` runs llama.cpp (GPU and CPU) and both antfly variants against
Gemma 4 E4B, Gemma 4 E4B QAT, and Qwen 3.6 35B-A3B. Results are written under
`results/`.

The Gemma 4 E4B QAT target uses the Hugging Face model
`unsloth/gemma-4-E4B-it-qat-GGUF` with
`gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`, matching the `UD-Q4_K_XL` llama.cpp
selector advertised by the model card.

The `antfly/` directory at the repo root is the local source checkout for the
latest variant. It's gitignored — clone `https://github.com/antflydb/antfly`
into `./antfly` (or symlink an existing checkout). `make build-antfly-latest`
(and therefore `make bench`) runs `git fetch && git checkout --force origin/main`
before building, so the binary always tracks canonical `antflydb/antfly` HEAD.
If `antfly/` isn't present, the latest variant is skipped; `make bench` will
still run llama.cpp and the brew stable.

## Overriding paths

The two antfly binaries can be overridden:

```bash
ANTFLY_STABLE_BIN=/custom/antfly ANTFLY_SRC=/path/to/antfly make bench
```

## Agent workflow

The intended use is to have an agent run the benchmark and post results to Zulip:

```
Run `make bench` and post results/gemma4-e4b.md to #eval-runs > termite-gemma-4
```

## Nightly automated benchmark

A system LaunchDaemon runs `scripts/nightly-bench.sh` at 2:05 AM Pacific
several nights a week. It builds the latest `antfly` from the antfly main
repo's `zig/` tree (cloned into `.nightly/antfly-src`, or reusing
`./antfly` if present) and benchmarks it alongside the brew-installed
stable `antfly` and llama.cpp.

The nightly runner:
- verifies the configured GGUF model is present and downloads it if needed
- builds the latest variant with `-Doptimize=ReleaseSafe -Dskip-openapi=true`,
  adding `-Dmlx=true` if `libmlxc.dylib` is present and falling back without
  MLX on build failure
- posts the report to `#eval-runs > termite-circus-nightly` on Zulip

The brew stable variant currently lacks the termite-zig `generate` subcommand,
so it's listed in the table with its version string but its perf cells stay
empty until a future release exposes `generate`.

The tracked plist template lives at:
```bash
launchd/com.termite-circus.nightly-bench.plist
```

Install or refresh it with:
```bash
sudo ./scripts/install-launchdaemon.sh
```

Manage it with:
```bash
# Check status
sudo launchctl print system/com.termite-circus.nightly-bench

# Stop
sudo launchctl bootout system /Library/LaunchDaemons/com.termite-circus.nightly-bench.plist

# Start
sudo launchctl bootstrap system /Library/LaunchDaemons/com.termite-circus.nightly-bench.plist

# Trigger immediately for a smoke test
sudo launchctl kickstart -k system/com.termite-circus.nightly-bench

# View logs
tail -f ~/.termite-circus-nightly.log

# Manual run (without Zulip post)
./scripts/nightly-bench.sh

# One-off run at a specific local time
sudo ./scripts/schedule-launchdaemon-once.sh YYYY-MM-DD HH:MM
```

Notes:
- `LaunchDaemon` is the right choice when you want to install and manage the
  job entirely from a background shell with `sudo`.
- If the machine is asleep at 2:05 AM, the run is still missed. Add a
  scheduled wake with `pmset repeat` if that matters.
- Override `ANTFLY_SRC_DIR` in `.env` to point at a different antfly checkout
  (defaults to `./antfly` if present, otherwise `.nightly/antfly-src`).
- Override `ANTFLY_STABLE_BIN` in `.env` to point at a non-brew `antfly`.
- Override `MODEL_REPO` or `MODEL_FILE` in `.env` to use a different GGUF
  benchmark model.
- For the Gemma 4 E4B QAT nightly candidate, set
  `MODEL_REPO=unsloth/gemma-4-E4B-it-qat-GGUF` and
  `MODEL_FILE=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`.
- Override `MLX_ROOT` in `.env` if `libmlxc.dylib` is not under
  `$HOMEBREW_PREFIX` or `~/.homebrew`.
