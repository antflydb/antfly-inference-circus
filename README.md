<img width="2304" height="1848" alt="termite-circus" src="https://github.com/user-attachments/assets/f1feba1e-a4bf-4a37-a17d-b2c57389b10f" />

# termite-circus

Benchmark suite for [termite](https://github.com/antflydb/termite) (our local ML inference engine) vs llama.cpp.

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
Run `make bench` in ~/rowan/termite-circus and post results/gemma4-e4b.md
to #eval-runs > termite-gemma-4
```

For benchmarking code changes, ask the agent to make changes in `termite-zig-experimentation/` first, then run the benchmark. The output will include both the clean baseline and the experimental build so the diff is obvious.
