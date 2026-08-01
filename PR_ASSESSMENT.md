# Open PR assessment (2026-07-31)

The replacement harness was based on published `main` at `23ddbb1`. Neither
open pull request should be merged as a whole; both target the retired Go,
launchd, nightly, and Zulip-oriented workflow.

## PR #3 — Gemma 4 E4B QAT benchmark

Still relevant:

- retain Gemma 4 E4B QAT q4_0 as a full-profile cell;
- keep the Antfly MTP/draft-model configuration visible and distinct from
  reference-provider rows;
- measure a complete rooted process tree and account for Metal allocations.

Superseded:

- the rc.17 stable-binary assumption;
- the Go-only generator harness and fixed nightly model list;
- scheduled launchd/Zulip publication;
- adding RSS and wired growth, which can double-count model pages. The current
  harness reports the maximum of rooted-tree RSS and system demand growth.

## PR #4 — rename to Antfly inference

Still relevant:

- use the `antfly inference` CLI and `~/.antfly/inference/models` layout;
- label the product-facing row “Antfly inference”;
- sum RSS across the complete rooted process tree.

Superseded:

- CLI fallback probing and the old Markdown table parser;
- renamed launchd and Zulip automation;
- the generator-only Go harness.

The relevant behavior from both PRs is represented in `matrix.json`,
`scripts/circus.py`, and the parent antfly-circus JSON consumer. Closing or
otherwise changing the PRs is intentionally a separate GitHub action.

## Repin order

The parent repository's broken gitlink can be repaired to published main
`23ddbb1` immediately. The durable final pin for this implementation must be a
new termite-circus commit that exists on its remote; only then should the parent
gitlink advance to that commit. Pointing the parent at an unpublished local
submodule commit would recreate the original “not our ref” failure.
