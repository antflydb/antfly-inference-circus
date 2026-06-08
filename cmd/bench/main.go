package main

import (
	"bytes"
	"context"
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
	modelDir  = flag.String("model-dir", "", "Path to termite model directory")
	llamaBin  = flag.String("llama-cli", "llama-cli", "Path to llama-cli binary")

	antflyStableBin  = flag.String("antfly-stable", "", "Path to stable `antfly` binary (e.g. brew-installed)")
	antflyStableDir  = flag.String("antfly-stable-dir", "", "Source dir for stable antfly version reporting (optional)")
	antflyStableName = flag.String("antfly-stable-name", "antfly (brew stable)", "Display name for stable antfly")
	antflyLatestBin  = flag.String("antfly-latest", "", "Path to latest `antfly` binary built from antfly/zig")
	antflyLatestDir  = flag.String("antfly-latest-dir", "", "Path to antfly source repo (for git hash)")
	antflyLatestName = flag.String("antfly-latest-name", "antfly (latest zig)", "Display name for latest antfly")

	prompt     = flag.String("prompt", "Explain what a compiler does in two sentences.", "Prompt")
	promptFile = flag.String("prompt-file", "", "Path to newline-delimited prompts")
	maxTokens  = flag.Int("n", 32, "Max tokens to generate")
	output     = flag.String("output", "results/gemma4-e4b.md", "Output markdown file")
)

const promptPlaceholder = "__BENCH_PROMPT__"

type benchConfig struct {
	Name   string
	Cmd    string
	Args   []string
	UsePTY bool
	Commit string // short git hash (for clean repos)
}

type benchResult struct {
	Config    benchConfig
	Prompt    string
	GenTPS    float64
	PromptTPS float64
	PrefillMs float64
	PeakRSSMB float64
	WallSec   float64
	TokensGen int
	RawOutput string
	Error     string
}

type benchSummary struct {
	Config    benchConfig
	GenTPS    float64
	PrefillMs float64
	PeakRSSMB float64
	WallSec   float64
	TokensGen int
	RawOutput string
	Error     string
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// generateSubcommand returns the subcommand chain that gets to `generate` for
// the given antfly binary. It tries `inference generate` (latest antfly), then
// `termite generate` (transitional), then bare `generate` (legacy termite-zig).
// Returns nil if no shape works.
func generateSubcommand(bin string) []string {
	candidates := [][]string{
		{"inference", "generate"},
		{"termite", "generate"},
		{"generate"},
	}
	for _, prefix := range candidates {
		probe := append(append([]string{}, prefix...), "--help")
		if err := exec.Command(bin, probe...).Run(); err == nil {
			return prefix
		}
	}
	return nil
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

func argsForPrompt(args []string, prompt string) []string {
	resolved := make([]string, len(args))
	for i, arg := range args {
		if arg == promptPlaceholder {
			resolved[i] = prompt
			continue
		}
		resolved[i] = arg
	}
	return resolved
}

func summarizeRuns(cfg benchConfig, runs []benchResult) benchSummary {
	summary := benchSummary{Config: cfg}
	if len(runs) == 0 {
		summary.Error = "no benchmark runs"
		return summary
	}

	var genSum, prefillSum, peakSum, wallSum float64
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

	n := strconv.Itoa(*maxTokens)

	llamaBase := []string{
		"-m", *modelGGUF, "-p", promptPlaceholder, "-n", n,
		"--temp", "0", "--no-display-prompt", "--perf",
		"--single-turn", "--reasoning-format", "none",
	}

	termiteArgs := func(prefix []string) []string {
		args := append([]string{}, prefix...)
		return append(args,
			*modelDir, promptPlaceholder,
			"--backend", "mlx", "--max-tokens", n,
			"--backend-budget-mb", "20480",
			"--temperature", "0",
			"--print-timing", "--print-token-count", "--print-finish-reason",
		)
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

	// antfly (brew stable) — probe for any generate subcommand shape.
	if *antflyStableBin != "" && fileExists(*antflyStableBin) {
		cfg := benchConfig{
			Name:   *antflyStableName,
			Cmd:    *antflyStableBin,
			Commit: gitHash(*antflyStableDir),
		}
		if prefix := generateSubcommand(*antflyStableBin); prefix != nil {
			cfg.Args = termiteArgs(prefix)
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
			cfg.Args = termiteArgs(prefix)
		}
		configs = append(configs, cfg)
	}

	var results []benchSummary
	for _, cfg := range configs {
		var runs []benchResult
		for idx, benchPrompt := range prompts {
			fmt.Fprintf(os.Stderr, "running: %s", cfg.Name)
			if cfg.Commit != "" {
				fmt.Fprintf(os.Stderr, " (%s)", cfg.Commit)
			}
			fmt.Fprintf(os.Stderr, " [prompt %d/%d] ...\n", idx+1, len(prompts))

			var r benchResult
			if cfg.Args == nil {
				r = benchResult{
					Config: cfg,
					Prompt: benchPrompt,
					Error:  "no compatible `generate` subcommand (tried `inference generate`, `termite generate`, `generate`); GGUF benchmark not supported by this build",
				}
			} else {
				promptCfg := cfg
				promptCfg.Args = argsForPrompt(cfg.Args, benchPrompt)
				r = runBench(promptCfg)
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
		results = append(results, summarizeRuns(cfg, runs))
	}

	if err := writeMarkdown(results, prompts); err != nil {
		fmt.Fprintf(os.Stderr, "error writing output: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", *output)
}

func runBench(cfg benchConfig) benchResult {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
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

	start := time.Now()
	if err := cmd.Start(); err != nil {
		return benchResult{Config: cfg, Error: fmt.Sprintf("start: %v", err)}
	}

	var peakRSSKB int64
	done := make(chan struct{})
	go pollRSS(cmd.Process.Pid, &peakRSSKB, done)

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
		}
		if !strings.Contains(cfg.Name, "llama") && termiteRunCompleted(raw) {
			r.Error = ""
			r.TokensGen = *maxTokens
			parseTermite(&r)
		}
		return r
	}

	r := benchResult{
		Config:    cfg,
		RawOutput: raw,
		WallSec:   wall.Seconds(),
		PeakRSSMB: float64(peakRSSKB) / 1024,
		TokensGen: *maxTokens,
	}

	if strings.Contains(cfg.Name, "llama") {
		parseLlama(&r)
	} else {
		parseTermite(&r)
	}
	return r
}

// ─── Parsers ─────────────────────────────────────────────────────

var (
	reTermiteTiming = regexp.MustCompile(`generate_timing_ms:.*?prefill=(\d+).*?decode=(\d+).*?total=(\d+)`)
	reTermiteTotal  = regexp.MustCompile(`timing_ms:.*?total=(\d+)`)
	reTermiteTokens = regexp.MustCompile(`tokens=(\d+)`)
	reLlamaPerf     = regexp.MustCompile(`Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s`)
	reANSI          = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
)

func termiteRunCompleted(raw string) bool {
	return reTermiteTiming.MatchString(raw) &&
		reTermiteTokens.MatchString(raw) &&
		strings.Contains(raw, "finish_reason=")
}

func parseTermite(r *benchResult) {
	if m := reTermiteTiming.FindStringSubmatch(r.RawOutput); m != nil {
		prefill, _ := strconv.ParseFloat(m[1], 64)
		decode, _ := strconv.ParseFloat(m[2], 64)
		r.PrefillMs = prefill
		if m2 := reTermiteTokens.FindStringSubmatch(r.RawOutput); m2 != nil {
			r.TokensGen, _ = strconv.Atoi(m2[1])
		}
		if decode > 0 && r.TokensGen > 0 {
			r.GenTPS = float64(r.TokensGen) / (decode / 1000)
		}
	}
	if m := reTermiteTotal.FindStringSubmatch(r.RawOutput); m != nil {
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
			rss := maxTreeRSS(pid)
			for {
				old := atomic.LoadInt64(peakKB)
				if rss <= old || atomic.CompareAndSwapInt64(peakKB, old, rss) {
					break
				}
			}
		}
	}
}

func maxTreeRSS(pid int) int64 {
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
	var maxRSS int64
	for _, p := range procs {
		if tree[p.pid] && p.rss > maxRSS {
			maxRSS = p.rss
		}
	}
	return maxRSS
}

// ─── Markdown output ─────────────────────────────────────────────

func writeMarkdown(results []benchSummary, prompts []string) error {
	var b strings.Builder

	modelName := filepath.Base(filepath.Dir(*modelGGUF)) + "/" + filepath.Base(*modelGGUF)
	hostname, _ := os.Hostname()

	b.WriteString("# Termite vs llama.cpp — Gemma 4 E4B Benchmark\n\n")
	b.WriteString(fmt.Sprintf("**Date:** %s  \n", time.Now().Format("2006-01-02 15:04")))
	b.WriteString(fmt.Sprintf("**Host:** %s (%s)  \n", hostname, archString()))
	b.WriteString(fmt.Sprintf("**Model:** `%s`  \n", modelName))
	if len(prompts) == 1 {
		b.WriteString(fmt.Sprintf("**Prompt:** %q  \n", prompts[0]))
	} else {
		b.WriteString(fmt.Sprintf("**Prompts:** %d nightly bug questions  \n", len(prompts)))
	}
	b.WriteString(fmt.Sprintf("**Max tokens:** %d  \n\n", *maxTokens))

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
	b.WriteString("- Prefill (ms) for termite = reported prefill time (excludes model load)\n")
	b.WriteString("- Peak RSS measured via `ps` polling at 200ms intervals\n\n")

	// Table
	b.WriteString("| Engine | Gen (tok/s) | Prefill (ms) | Peak RSS (MB) | Wall (s) | Tokens |\n")
	b.WriteString("|--------|-------------|--------------|---------------|----------|--------|\n")
	for _, r := range results {
		if r.Error != "" {
			b.WriteString(fmt.Sprintf("| %s | — | — | — | — | — |\n", r.Config.Name))
			continue
		}
		b.WriteString(fmt.Sprintf("| %s | %.1f | %.0f | %.0f | %.1f | %d |\n",
			r.Config.Name, r.GenTPS, r.PrefillMs, r.PeakRSSMB, r.WallSec, r.TokensGen))
	}
	b.WriteString("\n")

	// Per-run spoiler blocks
	for _, r := range results {
		b.WriteString(fmt.Sprintf("```spoiler %s\n", r.Config.Name))
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
