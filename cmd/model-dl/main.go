package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

func main() {
	repo := flag.String("repo", "", "HuggingFace repo (e.g. ggml-org/gemma-4-E4B-it-GGUF)")
	file := flag.String("file", "", "Filename to download (e.g. gemma-4-e4b-it-Q4_K_M.gguf)")
	flag.Parse()

	if *repo == "" || *file == "" {
		fmt.Fprintln(os.Stderr, "usage: model-dl -repo <org/repo> -file <filename>")
		os.Exit(1)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	destDir := filepath.Join(home, ".termite", "models", "generators", *repo)
	destPath := filepath.Join(destDir, *file)
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", *repo, *file)

	// Check remote size with HEAD request.
	remoteSize, err := headSize(url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error checking remote: %v\n", err)
		os.Exit(1)
	}

	// Skip if file already exists and matches remote size.
	if info, err := os.Stat(destPath); err == nil {
		if remoteSize > 0 && info.Size() == remoteSize {
			fmt.Printf("already exists: %s (%s)\n", destPath, humanBytes(info.Size()))
			return
		}
		// Size mismatch — attempt resume.
		fmt.Printf("partial file found (%s / %s), resuming...\n",
			humanBytes(info.Size()), humanBytes(remoteSize))
	}

	if err := ensureDir(destDir); err != nil {
		fmt.Fprintf(os.Stderr, "error creating directory: %v\n", err)
		os.Exit(1)
	}

	if err := download(url, destPath, remoteSize); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func ensureDir(path string) error {
	if info, err := os.Stat(path); err == nil {
		if info.IsDir() {
			return nil
		}
		return fmt.Errorf("%s exists and is not a directory", path)
	}

	if linkInfo, err := os.Lstat(path); err == nil && linkInfo.Mode()&os.ModeSymlink != 0 {
		if err := os.Remove(path); err != nil {
			return err
		}
	}

	return os.MkdirAll(path, 0o755)
}

func headSize(url string) (int64, error) {
	resp, err := http.Head(url)
	if err != nil {
		return 0, err
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusFound || resp.StatusCode == http.StatusMovedPermanently {
		// HuggingFace redirects — follow it.
		loc := resp.Header.Get("Location")
		if loc != "" {
			resp, err = http.Head(loc)
			if err != nil {
				return 0, err
			}
			resp.Body.Close()
		}
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HEAD %s: %s", url, resp.Status)
	}
	if cl := resp.Header.Get("Content-Length"); cl != "" {
		return strconv.ParseInt(cl, 10, 64)
	}
	return 0, nil
}

func download(url, destPath string, totalSize int64) error {
	// Check for existing partial file to resume.
	var existingSize int64
	if info, err := os.Stat(destPath); err == nil {
		existingSize = info.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil // follow redirects
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	// If server doesn't support Range, start from scratch.
	if existingSize > 0 && resp.StatusCode != http.StatusPartialContent {
		existingSize = 0
	}

	flags := os.O_CREATE | os.O_WRONLY
	if existingSize > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(destPath, flags, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	pw := &progressWriter{
		total:   totalSize,
		written: existingSize,
		start:   time.Now(),
	}

	fmt.Printf("downloading %s\n  → %s\n", url, destPath)
	if _, err := io.Copy(f, io.TeeReader(resp.Body, pw)); err != nil {
		return err
	}
	fmt.Println() // newline after progress
	fmt.Printf("done: %s\n", humanBytes(pw.written))
	return nil
}

type progressWriter struct {
	total   int64
	written int64
	start   time.Time
	last    time.Time
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n := len(p)
	pw.written += int64(n)

	now := time.Now()
	if now.Sub(pw.last) < 250*time.Millisecond {
		return n, nil
	}
	pw.last = now

	elapsed := now.Sub(pw.start).Seconds()
	rate := float64(pw.written) / elapsed / (1024 * 1024)

	if pw.total > 0 {
		pct := float64(pw.written) / float64(pw.total) * 100
		fmt.Fprintf(os.Stderr, "\r  %5.1f%%  %s / %s  (%.1f MB/s)",
			pct, humanBytes(pw.written), humanBytes(pw.total), rate)
	} else {
		fmt.Fprintf(os.Stderr, "\r  %s  (%.1f MB/s)", humanBytes(pw.written), rate)
	}
	return n, nil
}

func humanBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "model-dl — download GGUF models from HuggingFace Hub")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Usage:")
		fmt.Fprintln(os.Stderr, "  model-dl -repo <org/repo> -file <filename>")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Example:")
		fmt.Fprintln(os.Stderr, "  model-dl -repo ggml-org/gemma-4-E4B-it-GGUF -file gemma-4-e4b-it-Q4_K_M.gguf")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Downloads to ~/.termite/models/generators/<repo>/<file>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}

}
