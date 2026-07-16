# vhs script for the README demo GIF.
# Render with:  vhs docs/demo/watari.tape   (https://github.com/charmbracelet/vhs)
# Produces docs/demo/watari.gif. Requires a local Ollama with qwen3.5:4b and the
# eval corpus ingested (the ingest step below does that).

Output docs/demo/watari.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"

# --- Ingest some docs ---
Type "uv run watari ingest evals/corpora"
Enter
Sleep 4s

# --- Ask a cited question ---
Type "uv run watari chat"
Enter
Sleep 3s
Type "What programming language is the Tidal greenhouse written in?"
Enter
Sleep 8s

# --- Show memory ---
Type "/remember"
Enter
Sleep 6s
Type "/exit"
Enter
Sleep 1s

# --- Run an eval suite ---
Type "uv run watari evals run --suite retrieval"
Enter
Sleep 12s
