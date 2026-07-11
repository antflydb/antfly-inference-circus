package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

var (
	modelGGUF = flag.String("model", "", "Path to GGUF file (for llama-cli)")
	modelDir  = flag.String("model-dir", "", "Path to Antfly inference model directory")
	llamaBin  = flag.String("llama-cli", "llama-cli", "Path to llama-cli binary")

	antflyStableBin  = flag.String("antfly-stable", "", "Path to stable `antfly` binary (e.g. brew-installed)")
	antflyStableDir  = flag.String("antfly-stable-dir", "", "Source dir for stable antfly version reporting (optional)")
	antflyStableName = flag.String("antfly-stable-name", "antfly (brew stable)", "Display name for stable antfly")
	antflyLatestBin  = flag.String("antfly-latest", "", "Path to latest `antfly` binary built from antfly/zig")
	antflyLatestDir  = flag.String("antfly-latest-dir", "", "Path to antfly source repo (for git hash)")
	antflyLatestName = flag.String("antfly-latest-name", "antfly (latest zig)", "Display name for latest antfly")

	prompt          = flag.String("prompt", "Explain what a compiler does in two sentences.", "Prompt")
	promptFile      = flag.String("prompt-file", "", "Path to newline-delimited prompts")
	maxTokens       = flag.Int("n", 32, "Max tokens to generate")
	tokenCounts     = flag.String("token-counts", "", "Comma-separated max-token values benchmarked per engine; overrides -n when set")
	output          = flag.String("output", "results/gemma4-e4b.md", "Output markdown file")
	backendBudgetMB = flag.Int("backend-budget-mb", 20480, "Antfly --backend-budget-mb value")

	cooldownSeconds   = flag.Int("cooldown-seconds", 0, "Sleep this long before each measured run so samples start from a cooled, quiet host")
	runTimeoutSeconds = flag.Int("run-timeout-seconds", 120, "Per-run timeout in seconds")
	ignoreEOS         = flag.Bool("ignore-eos", false, "Keep generating past EOS so every run emits exactly the requested token count")

	antflyDraftModel        = flag.String("antfly-draft-model", "", "Gemma 4 MTP assistant model dir; enables speculative decoding on Antfly rows")
	antflySpeculativeK      = flag.Int("antfly-speculative-k", 2, "Antfly --speculative-k when a draft model is set")
	antflySpeculationPolicy = flag.String("antfly-speculation-policy", "force", "Antfly --speculation-policy when a draft model is set")
)

const (
	promptPlaceholder     = "__BENCH_PROMPT__"
	tokensPlaceholder     = "__BENCH_TOKENS__"
	timingJSONPlaceholder = "__BENCH_TIMING_JSON__"
)

type benchConfig struct {
	Name   string
	Cmd    string
	Args   []string
	UsePTY bool
	Commit string // short git hash (for clean repos)
}

type benchResult struct {
	Config       benchConfig
	Prompt       string
	GenTPS       float64
	PromptTPS    float64
	PrefillMs    float64
	PeakRSSMB    float64
	WiredDeltaMB float64
	WallSec      float64
	TokensGen    int
	RawOutput    string
	Error        string
}

type benchSummary struct {
	Config       benchConfig
	MaxTokens    int
	GenTPS       float64
	PrefillMs    float64
	PeakRSSMB    float64
	WiredDeltaMB float64
	WallSec      float64
	TokensGen    int
	MTP          string
	RawOutput    string
	Error        string
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// generateSubcommand returns the Antfly 0.2 inference generation command chain.
// It checks help text instead of exit status because the CLI prints help-on-error
// for invalid arguments.
func generateSubcommand(bin string) []string {
	parent := []string{"inference"}
	help := append(append([]string{}, parent...), "--help")
	out, _ := exec.Command(bin, help...).CombinedOutput()
	if reGenerateSubcmd.Match(out) {
		return append(parent, "generate")
	}
	return nil
}

// reGenerateSubcmd matches a line listing `generate` as a subcommand — e.g.
// "  generate    Run text generation" (cobra) or "  generate" (zig CLI).
var reGenerateSubcmd = regexp.MustCompile(`(?m)^\s+generate(\s|$)`)

// supportsJSONTiming reports whether `inference generate` accepts --json-timing.
// The speculative/MTP path omits the first_token_ms stdout line, so the JSON
// timing file is the only reliable prefill/decode split there.
func supportsJSONTiming(bin string) bool {
	out, _ := exec.Command(bin, "inference", "generate", "--help").CombinedOutput()
	return bytes.Contains(out, []byte("--json-timing"))
}

func gitHash(dir string) string {
	if dir == "" {
		return ""
	}
	out, err := exec.Command("git", "-C", dir, "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func loadPrompts() ([]string, error) {
	if *promptFile == "" {
		return []string{*prompt}, nil
	}
	raw, err := os.ReadFile(*promptFile)
	if err != nil {
		return nil, err
	}
	var prompts []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			prompts = append(prompts, line)
		}
	}
	if len(prompts) == 0 {
		return nil, fmt.Errorf("no prompts found in %s", *promptFile)
	}
	return prompts, nil
}

func argsForRun(args []string, prompt string, nTokens int, timingJSON string) []string {
	resolved := make([]string, len(args))
	for i, arg := range args {
		switch arg {
		case promptPlaceholder:
			resolved[i] = prompt
		case tokensPlaceholder:
			resolved[i] = strconv.Itoa(nTokens)
		case timingJSONPlaceholder:
			resolved[i] = timingJSON
		default:
			resolved[i] = arg
		}
	}
	return resolved
}

func parseTokenCounts() ([]int, error) {
	if strings.TrimSpace(*tokenCounts) == "" {
		return []int{*maxTokens}, nil
	}
	var counts []int
	for _, field := range strings.Split(*tokenCounts, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		n, err := strconv.Atoi(field)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("invalid -token-counts entry %q", field)
		}
		counts = append(counts, n)
	}
	if len(counts) == 0 {
		return nil, fmt.Errorf("no token counts found in %q", *tokenCounts)
	}
	return counts, nil
}

func summarizeRuns(cfg benchConfig, nTokens int, runs []benchResult) benchSummary {
	summary := benchSummary{Config: cfg, MaxTokens: nTokens, MTP: "-"}
	if len(runs) == 0 {
		summary.Error = "no benchmark runs"
		return summary
	}

	isAntfly := !strings.Contains(cfg.Name, "llama")
	if isAntfly && *antflyDraftModel != "" && cfg.Args != nil {
		summary.MTP = "off"
		for _, run := range runs {
			if run.Error == "" && reAntflyMTPEnabled.MatchString(run.RawOutput) {
				summary.MTP = "on"
				break
			}
		}
	}

	var genSum, prefillSum, peakSum, wiredSum, wallSum float64
	var successCount int
	var raw strings.Builder

	for _, run := range runs {
		raw.WriteString("## Prompt: ")
		raw.WriteString(run.Prompt)
		raw.WriteString("\n")
		if run.Error != "" {
			raw.WriteString("ERROR: ")
			raw.WriteString(run.Error)
			raw.WriteString("\n")
		}
		if run.RawOutput != "" {
			raw.WriteString(strings.TrimSpace(run.RawOutput))
			raw.WriteString("\n")
		}
		raw.WriteString("\n")

		if run.Error != "" {
			continue
		}
		successCount++
		genSum += run.GenTPS
		prefillSum += run.PrefillMs
		peakSum += run.PeakRSSMB
		wiredSum += run.WiredDeltaMB
		wallSum += run.WallSec
		summary.TokensGen += run.TokensGen
	}

	summary.RawOutput = strings.TrimSpace(raw.String())
	if successCount == 0 {
		summary.Error = "all prompt runs failed"
		return summary
	}

	summary.GenTPS = genSum / float64(successCount)
	summary.PrefillMs = prefillSum / float64(successCount)
	summary.PeakRSSMB = peakSum / float64(successCount)
	summary.WiredDeltaMB = wiredSum / float64(successCount)
	summary.WallSec = wallSum / float64(successCount)
	summary.TokensGen /= successCount
	return summary
}

func main() {
	flag.Parse()
	if *modelGGUF == "" || *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: bench -model <gguf> -model-dir <dir> [engine flags]")
		os.Exit(1)
	}
	if !fileExists(*modelGGUF) {
		fmt.Fprintf(os.Stderr, "model GGUF not found: %s\n", *modelGGUF)
		os.Exit(1)
	}
	if info, err := os.Stat(*modelDir); err != nil || !info.IsDir() {
		fmt.Fprintf(os.Stderr, "model directory not found: %s\n", *modelDir)
		os.Exit(1)
	}
	prompts, err := loadPrompts()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading prompts: %v\n", err)
		os.Exit(1)
	}
	tokenList, err := parseTokenCounts()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	llamaBase := []string{
		"-m", *modelGGUF, "-p", promptPlaceholder, "-n", tokensPlaceholder,
		"--temp", "0", "--no-display-prompt", "--perf",
		"--single-turn", "--reasoning-format", "none",
	}
	if *ignoreEOS {
		llamaBase = append(llamaBase, "--ignore-eos")
	}

	antflyInferenceArgs := func(prefix []string, jsonTiming bool) []string {
		args := append([]string{}, prefix...)
		args = append(args,
			*modelDir, promptPlaceholder,
			"--backend", "metal", "--max-tokens", tokensPlaceholder,
			"--host-budget-mb", strconv.Itoa(*backendBudgetMB),
			"--backend-budget-mb", strconv.Itoa(*backendBudgetMB),
			"--combined-budget-mb", strconv.Itoa(*backendBudgetMB),
			"--temperature", "0",
			"--print-timing", "--print-token-count", "--print-finish-reason",
		)
		if jsonTiming {
			args = append(args, "--json-timing", timingJSONPlaceholder)
		}
		if *ignoreEOS {
			args = append(args, "--ignore-eos")
		}
		if *antflyDraftModel != "" {
			args = append(args,
				"--draft-model", *antflyDraftModel,
				"--speculative-k", strconv.Itoa(*antflySpeculativeK),
				"--speculation-policy", *antflySpeculationPolicy,
			)
		}
		return args
	}

	var configs []benchConfig

	// llama.cpp GPU
	configs = append(configs, benchConfig{
		Name: "llama.cpp (GPU)", Cmd: *llamaBin,
		Args: append(llamaBase, "-ngl", "99"), UsePTY: true,
	})

	// llama.cpp CPU
	configs = append(configs, benchConfig{
		Name: "llama.cpp (CPU)", Cmd: *llamaBin,
		Args: append(append([]string{}, llamaBase...), "-ngl", "0"), UsePTY: true,
	})

	// antfly (brew stable) — probe for the Antfly 0.2 inference command shape.
	if *antflyStableBin != "" && fileExists(*antflyStableBin) {
		cfg := benchConfig{
			Name:   *antflyStableName,
			Cmd:    *antflyStableBin,
			Commit: gitHash(*antflyStableDir),
		}
		if prefix := generateSubcommand(*antflyStableBin); prefix != nil {
			cfg.Args = antflyInferenceArgs(prefix, supportsJSONTiming(*antflyStableBin))
		}
		configs = append(configs, cfg)
	}

	// antfly (latest, built from antfly/zig).
	if *antflyLatestBin != "" && fileExists(*antflyLatestBin) {
		cfg := benchConfig{
			Name:   *antflyLatestName,
			Cmd:    *antflyLatestBin,
			Commit: gitHash(*antflyLatestDir),
		}
		if prefix := generateSubcommand(*antflyLatestBin); prefix != nil {
			cfg.Args = antflyInferenceArgs(prefix, supportsJSONTiming(*antflyLatestBin))
		}
		configs = append(configs, cfg)
	}

	var results []benchSummary
	for _, cfg := range configs {
		for _, nTokens := range tokenList {
			var runs []benchResult
			for idx, benchPrompt := range prompts {
				if *cooldownSeconds > 0 && cfg.Args != nil {
					fmt.Fprintf(os.Stderr, "cooldown: %ds before %s (n=%d)\n", *cooldownSeconds, cfg.Name, nTokens)
					time.Sleep(time.Duration(*cooldownSeconds) * time.Second)
				}
				fmt.Fprintf(os.Stderr, "running: %s", cfg.Name)
				if cfg.Commit != "" {
					fmt.Fprintf(os.Stderr, " (%s)", cfg.Commit)
				}
				fmt.Fprintf(os.Stderr, " [n=%d, prompt %d/%d] ...\n", nTokens, idx+1, len(prompts))

				var r benchResult
				if cfg.Args == nil {
					r = benchResult{
						Config: cfg,
						Prompt: benchPrompt,
						Error:  "no compatible `inference generate` subcommand; GGUF benchmark not supported by this build",
					}
				} else {
					timingJSON := filepath.Join(os.TempDir(),
						fmt.Sprintf("antfly-bench-timing-%d-%d-%d.json", os.Getpid(), nTokens, idx))
					os.Remove(timingJSON)
					promptCfg := cfg
					promptCfg.Args = argsForRun(cfg.Args, benchPrompt, nTokens, timingJSON)
					r = runBench(promptCfg, nTokens, timingJSON)
					os.Remove(timingJSON)
					r.Config = cfg
					r.Prompt = benchPrompt
				}

				if r.Error != "" {
					fmt.Fprintf(os.Stderr, "  ERROR: %s\n", r.Error)
				} else {
					fmt.Fprintf(os.Stderr, "  gen: %.1f tok/s | prefill: %.0fms | RSS: %.0f MB | wall: %.1fs\n",
						r.GenTPS, r.PrefillMs, r.PeakRSSMB, r.WallSec)
				}
				runs = append(runs, r)
			}
			results = append(results, summarizeRuns(cfg, nTokens, runs))
		}
	}

	if err := writeMarkdown(results, prompts, tokenList); err != nil {
		fmt.Fprintf(os.Stderr, "error writing output: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *output)
}

func runBench(cfg benchConfig, nTokens int, timingJSON string) benchResult {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*runTimeoutSeconds)*time.Second)
	defer cancel()

	var cmd *exec.Cmd
	if cfg.UsePTY && runtime.GOOS == "darwin" {
		scriptArgs := []string{"-q", "/dev/null", cfg.Cmd}
		scriptArgs = append(scriptArgs, cfg.Args...)
		cmd = exec.CommandContext(ctx, "script", scriptArgs...)
	} else {
		cmd = exec.CommandContext(ctx, cfg.Cmd, cfg.Args...)
	}

	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined

	baselineWiredKB := wiredKB()
	start := time.Now()
	if err := cmd.Start(); err != nil {
		return benchResult{Config: cfg, Error: fmt.Sprintf("start: %v", err)}
	}

	var peakRSSKB, peakWiredDeltaKB int64
	done := make(chan struct{})
	go pollRSS(cmd.Process.Pid, &peakRSSKB, done)
	go pollWiredDelta(baselineWiredKB, &peakWiredDeltaKB, done)

	err := cmd.Wait()
	close(done)
	wall := time.Since(start)

	raw := combined.String()
	raw = strings.ReplaceAll(raw, "\r", "")
	raw = reANSI.ReplaceAllString(raw, "")
	raw = strings.ReplaceAll(raw, "\x04", "")
	raw = strings.ReplaceAll(raw, "^D", "")

	if err != nil {
		r := benchResult{
			Config: cfg, RawOutput: raw,
			Error:   fmt.Sprintf("exit: %v", err),
			WallSec: wall.Seconds(), PeakRSSMB: float64(peakRSSKB) / 1024,
			WiredDeltaMB: float64(peakWiredDeltaKB) / 1024,
		}
		if !strings.Contains(cfg.Name, "llama") && antflyInferenceRunCompleted(raw) {
			r.Error = ""
			r.TokensGen = nTokens
			parseAntflyInference(&r)
			applyAntflyJSONTiming(&r, timingJSON)
		}
		return r
	}

	r := benchResult{
		Config:       cfg,
		RawOutput:    raw,
		WallSec:      wall.Seconds(),
		PeakRSSMB:    float64(peakRSSKB) / 1024,
		WiredDeltaMB: float64(peakWiredDeltaKB) / 1024,
		TokensGen:    nTokens,
	}

	if strings.Contains(cfg.Name, "llama") {
		parseLlama(&r)
	} else {
		parseAntflyInference(&r)
		applyAntflyJSONTiming(&r, timingJSON)
	}
	return r
}

// antflyJSONTiming mirrors the fields we need from `--json-timing` output.
// It is authoritative over the stdout regexes: the speculative/MTP path does
// not print first_token_ms, so JSON is the only prefill/decode split there.
type antflyJSONTiming struct {
	Tokens        int     `json:"tokens"`
	DecodeTokPerS float64 `json:"decode_tok_per_s"`
	TimingMs      struct {
		PrefillInner float64 `json:"prefill_inner"`
		DecodeInner  float64 `json:"decode_inner"`
		Total        float64 `json:"total"`
	} `json:"timing_ms"`
}

func applyAntflyJSONTiming(r *benchResult, path string) {
	if path == "" {
		return
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var t antflyJSONTiming
	if err := json.Unmarshal(raw, &t); err != nil {
		return
	}
	if t.Tokens > 0 {
		r.TokensGen = t.Tokens
	}
	if t.DecodeTokPerS > 0 {
		r.GenTPS = t.DecodeTokPerS
	}
	if t.TimingMs.PrefillInner > 0 {
		r.PrefillMs = t.TimingMs.PrefillInner
	}
	if t.TimingMs.Total > 0 {
		r.WallSec = t.TimingMs.Total / 1000
	}
}

// ─── Parsers ─────────────────────────────────────────────────────

var (
	reAntflyTimingOld  = regexp.MustCompile(`generate_timing_ms:.*?prefill=(\d+).*?decode=(\d+).*?total=(\d+)`)
	reAntflyTimingNew  = regexp.MustCompile(`(?m)^timing_ms:.*?generate=(\d+).*?total=(\d+)`)
	reAntflyFirstToken = regexp.MustCompile(`first_token_ms:.*?prefill=(\d+)`)
	reAntflyTotal      = regexp.MustCompile(`timing_ms:.*?total=(\d+)`)
	reAntflyTokens     = regexp.MustCompile(`tokens=(\d+)`)
	reAntflyMTPEnabled = regexp.MustCompile(`(?m)^speculative: .*mtp_enabled=true`)
	reLlamaPerf        = regexp.MustCompile(`Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s`)
	reANSI             = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
)

func antflyInferenceRunCompleted(raw string) bool {
	hasTiming := reAntflyTimingOld.MatchString(raw) || reAntflyTimingNew.MatchString(raw)
	return hasTiming &&
		reAntflyTokens.MatchString(raw) &&
		strings.Contains(raw, "finish_reason=")
}

func parseAntflyInference(r *benchResult) {
	if m2 := reAntflyTokens.FindStringSubmatch(r.RawOutput); m2 != nil {
		r.TokensGen, _ = strconv.Atoi(m2[1])
	}
	if m := reAntflyTimingOld.FindStringSubmatch(r.RawOutput); m != nil {
		prefill, _ := strconv.ParseFloat(m[1], 64)
		decode, _ := strconv.ParseFloat(m[2], 64)
		r.PrefillMs = prefill
		if decode > 0 && r.TokensGen > 0 {
			r.GenTPS = float64(r.TokensGen) / (decode / 1000)
		}
	} else if m := reAntflyTimingNew.FindStringSubmatch(r.RawOutput); m != nil {
		generate, _ := strconv.ParseFloat(m[1], 64)
		var prefill float64
		if fm := reAntflyFirstToken.FindStringSubmatch(r.RawOutput); fm != nil {
			prefill, _ = strconv.ParseFloat(fm[1], 64)
			r.PrefillMs = prefill
		}
		decode := generate - prefill
		if decode > 0 && r.TokensGen > 0 {
			r.GenTPS = float64(r.TokensGen) / (decode / 1000)
		}
	}
	if m := reAntflyTotal.FindStringSubmatch(r.RawOutput); m != nil {
		total, _ := strconv.ParseFloat(m[1], 64)
		if total > 0 {
			r.WallSec = total / 1000
		}
	}
}

func parseLlama(r *benchResult) {
	if m := reLlamaPerf.FindStringSubmatch(r.RawOutput); m != nil {
		r.PromptTPS, _ = strconv.ParseFloat(m[1], 64)
		r.GenTPS, _ = strconv.ParseFloat(m[2], 64)
		if r.GenTPS > 0 {
			genSec := float64(r.TokensGen) / r.GenTPS
			r.PrefillMs = (r.WallSec - genSec) * 1000
			if r.PrefillMs < 0 {
				r.PrefillMs = 0
			}
		}
	}
}

// ─── RSS polling ─────────────────────────────────────────────────

func pollRSS(pid int, peakKB *int64, done <-chan struct{}) {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			rss := sumTreeRSS(pid)
			for {
				old := atomic.LoadInt64(peakKB)
				if rss <= old || atomic.CompareAndSwapInt64(peakKB, old, rss) {
					break
				}
			}
		}
	}
}

// pollWiredDelta tracks the peak growth of system-wide wired memory over a
// baseline taken just before the benchmarked process starts. Metal/IOKit
// buffers (where GPU engines keep model weights) are wired and charged to
// the system rather than the allocating task, so process RSS misses them.
// Requires a quiet host; the per-run cooldown keeps this usable.
func pollWiredDelta(baselineKB int64, peakDeltaKB *int64, done <-chan struct{}) {
	if baselineKB <= 0 {
		return
	}
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			delta := wiredKB() - baselineKB
			if delta <= 0 {
				continue
			}
			for {
				old := atomic.LoadInt64(peakDeltaKB)
				if delta <= old || atomic.CompareAndSwapInt64(peakDeltaKB, old, delta) {
					break
				}
			}
		}
	}
}

var (
	reVMStatPageSize = regexp.MustCompile(`page size of (\d+) bytes`)
	reVMStatWired    = regexp.MustCompile(`Pages wired down:\s+(\d+)\.`)
)

func wiredKB() int64 {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}
	pageSize := int64(16384)
	if m := reVMStatPageSize.FindSubmatch(out); m != nil {
		if n, err := strconv.ParseInt(string(m[1]), 10, 64); err == nil && n > 0 {
			pageSize = n
		}
	}
	m := reVMStatWired.FindSubmatch(out)
	if m == nil {
		return 0
	}
	pages, err := strconv.ParseInt(string(m[1]), 10, 64)
	if err != nil {
		return 0
	}
	return pages * pageSize / 1024
}

func sumTreeRSS(pid int) int64 {
	out, err := exec.Command("ps", "-eo", "pid,ppid,rss").Output()
	if err != nil {
		return 0
	}
	type proc struct {
		pid, ppid int
		rss       int64
	}
	var procs []proc
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		p, _ := strconv.Atoi(fields[0])
		pp, _ := strconv.Atoi(fields[1])
		r, _ := strconv.ParseInt(fields[2], 10, 64)
		if p > 0 {
			procs = append(procs, proc{p, pp, r})
		}
	}
	tree := map[int]bool{pid: true}
	changed := true
	for changed {
		changed = false
		for _, p := range procs {
			if tree[p.ppid] && !tree[p.pid] {
				tree[p.pid] = true
				changed = true
			}
		}
	}
	var sumRSS int64
	for _, p := range procs {
		if tree[p.pid] {
			sumRSS += p.rss
		}
	}
	return sumRSS
}

// ─── Markdown output ─────────────────────────────────────────────

func writeMarkdown(results []benchSummary, prompts []string, tokenList []int) error {
	var b strings.Builder

	modelName := filepath.Base(filepath.Dir(*modelGGUF)) + "/" + filepath.Base(*modelGGUF)
	hostname, _ := os.Hostname()

	b.WriteString(fmt.Sprintf("# Antfly inference vs llama.cpp — %s\n\n", filepath.Base(*modelGGUF)))
	b.WriteString(fmt.Sprintf("**Date:** %s  \n", time.Now().Format("2006-01-02 15:04")))
	b.WriteString(fmt.Sprintf("**Host:** %s (%s)  \n", hostname, archString()))
	b.WriteString(fmt.Sprintf("**Model:** `%s`  \n", modelName))
	if len(prompts) == 1 {
		b.WriteString(fmt.Sprintf("**Prompt:** %q  \n", prompts[0]))
	} else {
		b.WriteString(fmt.Sprintf("**Prompts:** %d nightly bug questions  \n", len(prompts)))
	}
	tokenStrs := make([]string, len(tokenList))
	for i, n := range tokenList {
		tokenStrs[i] = strconv.Itoa(n)
	}
	b.WriteString(fmt.Sprintf("**Max tokens:** %s  \n", strings.Join(tokenStrs, ", ")))
	if *cooldownSeconds > 0 {
		b.WriteString(fmt.Sprintf("**Cooldown:** %ds before each measured run (single cooled samples)  \n", *cooldownSeconds))
	}
	if *ignoreEOS {
		b.WriteString("**Ignore EOS:** generation continues to the full token budget on both engines  \n")
	}
	if *antflyDraftModel != "" {
		b.WriteString(fmt.Sprintf("**MTP:** draft model `%s`, speculative-k %d, speculation-policy %s  \n",
			filepath.Base(*antflyDraftModel), *antflySpeculativeK, *antflySpeculationPolicy))
	}
	b.WriteString("\n")

	if len(prompts) > 1 {
		for i, benchPrompt := range prompts {
			b.WriteString(fmt.Sprintf("%d. %s\n", i+1, benchPrompt))
		}
		b.WriteString("\n")
	}

	// Engine versions
	b.WriteString("**Versions:**\n")
	for _, r := range results {
		if r.Config.Commit != "" {
			b.WriteString(fmt.Sprintf("- %s: `%s`\n", r.Config.Name, r.Config.Commit))
		}
	}
	b.WriteString("\n")

	b.WriteString("**Notes:**\n")
	b.WriteString("- Prefill (ms) for llama.cpp = wall time minus generation time (includes model load)\n")
	b.WriteString("- Prefill (ms) for Antfly inference = reported prefill time (excludes model load)\n")
	b.WriteString("- Peak Mem = process-tree RSS (via `ps` polling at 200ms intervals, summed across client + spawned backends) plus the peak growth of system-wide wired memory over a pre-launch baseline. The wired component captures Metal/IOKit model buffers that macOS charges to the system instead of the process; it requires a quiet host. Per-engine breakdown below the table.\n\n")

	// Table
	b.WriteString("| Engine | Gen (tok/s) | Prefill (ms) | Peak Mem (MB) | Wall (s) | Tokens | MTP |\n")
	b.WriteString("|--------|-------------|--------------|---------------|----------|--------|-----|\n")
	for _, r := range results {
		if r.Error != "" {
			b.WriteString(fmt.Sprintf("| %s | — | — | — | — | — | %s |\n", r.Config.Name, r.MTP))
			continue
		}
		b.WriteString(fmt.Sprintf("| %s | %.3f | %.0f | %.0f | %.1f | %d | %s |\n",
			r.Config.Name, r.GenTPS, r.PrefillMs, r.PeakRSSMB+r.WiredDeltaMB, r.WallSec, r.TokensGen, r.MTP))
	}
	b.WriteString("\nMemory breakdown (process-tree RSS + wired Δ, MB):\n")
	for _, r := range results {
		if r.Error != "" {
			continue
		}
		b.WriteString(fmt.Sprintf("- %s: %.0f + %.0f\n", r.Config.Name, r.PeakRSSMB, r.WiredDeltaMB))
	}
	b.WriteString("\n")

	// Per-run spoiler blocks
	for _, r := range results {
		label := r.Config.Name
		if len(tokenList) > 1 {
			label = fmt.Sprintf("%s (n=%d)", r.Config.Name, r.MaxTokens)
		}
		b.WriteString(fmt.Sprintf("```spoiler %s\n", label))
		if r.Error != "" {
			b.WriteString(fmt.Sprintf("ERROR: %s\n", r.Error))
		}
		if r.RawOutput != "" {
			b.WriteString(strings.TrimSpace(r.RawOutput))
			b.WriteString("\n")
		}
		b.WriteString("```\n\n")
	}

	dir := *output
	if i := strings.LastIndex(dir, "/"); i >= 0 {
		os.MkdirAll(dir[:i], 0o755)
	}
	return os.WriteFile(*output, []byte(b.String()), 0o644)
}

func archString() string {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err == nil {
		return strings.TrimSpace(string(out))
	}
	return "unknown"
}
