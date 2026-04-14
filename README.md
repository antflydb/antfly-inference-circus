<img width="2304" height="1848" alt="termite-circus" src="https://github.com/user-attachments/assets/f1feba1e-a4bf-4a37-a17d-b2c57389b10f" />

# termite-circus

Benchmark suite for [termite](https://github.com/antflydb/termite) (Antfly's local ML inference engine) vs llama.cpp.

## Quick start

```bash
git submodule update --init
make download-gemma4-e4b   # ~5 GB GGUF, skips if already present
make bench                 # build + benchmark
```

`make bench` runs llama.cpp (GPU and CPU) and termite-zig-clean against Gemma 4 E4B, then writes results to `results/gemma4-e4b.md`.

## Experimenting

To benchmark local changes to termite-zig, clone or symlink a working copy into `termite-zig-experimentation/`:

```bash
git clone https://github.com/antflydb/termite-zig termite-zig-experimentation
# hack on it...
make bench   # automatically picks it up if present
```

That directory is gitignored. When it exists, `make bench` builds and benchmarks it alongside the clean baseline. When it doesn't, it's silently skipped.

## Agent workflow

The intended use is to have an agent run the benchmark and post results to Zulip:

```
Run `make bench` and post results/gemma4-e4b.md to #eval-runs > termite-gemma-4
```

For benchmarking code changes, ask the agent to make changes in `termite-zig-experimentation/` first, then run the benchmark. The output will include both the clean baseline and the experimental build so the diff is obvious.

## Nightly automated benchmark

A launchd job runs `scripts/nightly-bench.sh` at 2:05 AM Pacific every night. It builds termite-zig at both the latest tagged release and tip of main, benchmarks both against llama.cpp, and posts results to `#eval-runs > termite-circus-nightly` on Zulip.

The job is defined at:
```
~/Library/LaunchAgents/com.termite-circus.nightly-bench.plist
```

Manage it with:
```bash
# Check status
launchctl list | grep termite

# Stop
launchctl unload ~/Library/LaunchAgents/com.termite-circus.nightly-bench.plist

# Start
launchctl load -w ~/Library/LaunchAgents/com.termite-circus.nightly-bench.plist

# View logs
tail -f ~/.termite-circus-nightly.log

# Manual run (without Zulip post)
./scripts/nightly-bench.sh
```
