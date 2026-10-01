# Rushing Attempts Model v3

## Live source
The dashboard uses The Odds API directly.

Primary line: PrizePicks standard rushing-attempt line.
Fallback: median of DraftKings, FanDuel, BetMGM and BetRivers.

## Setup on Mac

1. Get an API key from The Odds API.
2. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
3. Paste your key in that file.
4. Run:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Automatic weekly rollover
The app checks nflverse schedule/results every refresh. Once the current week is final and weekly player stats are available, it archives results, builds the next week's original 50/50 ranking, and advances `state.json`.

## Persistent files
- historical_lines.csv
- results_history.csv
- rankings_history.csv
- state.json

Keep these files when updating the app.


## Patched live API build

This package is ready to run and includes the configured Streamlit secrets file.

Changes in this build:
- live lines cached for 10 minutes
- NFL event IDs cached for 10 minutes
- one-second spacing between event-level prop calls
- automatic 429 backoff: 2s, 4s, 8s, 12s
- clean rate-limit and exhausted-credit messages
- error handling that does not print the API request URL/key
- manual refresh button that clears the live-data cache
