# FPL 2025 Toolkit – CLI Reference

This document describes the unified command-line interface exposed by the package `fpl_weekly_updater`.

You can invoke the CLI via the Python module runner or via a packaged EXE (if built).

## Invocation

- Python module:
  - `python -m fpl_weekly_updater <command> [options]`
- EXE (after building):
  - `FPLWeeklyUpdater.exe <command> [options]`

Global verbosity flags available on all commands:
- `--log-level {CRITICAL|ERROR|WARNING|INFO|DEBUG}`
- `-q, --quiet` (alias for WARNING)
- `-v, --verbose` (alias for INFO)

The default console level is WARNING. You can also set `LOG_LEVEL` in the environment.

---

## Command: weekly

Run the main weekly update flow and generate the weekly PDF report.

Usage:
```bash
python -m fpl_weekly_updater weekly [--gameweek N] [--report-dir PATH] [--appendix] [--no-news] [-q|--quiet] [--log-level LEVEL]
```

Options:
- `--gameweek N` – For backtest-style runs, target a specific gameweek (triggers training where applicable).
- `--report-dir PATH` – Override the default report output directory (otherwise uses `REPORT_OUTPUT_DIR` or Desktop).
- `--appendix` – Also generate the weekly appendix variant (implemented).
- `--no-news` – Skip Perplexity/news analysis (uses cached data only or none).

Notes:
- Passwords are read securely from the OS keyring (see `set-password`).
- FPL status remains authoritative for selection/headings; Perplexity is narrative-only.

---

## Command: appendix

Generate only the appendix report (fixtures, differentials, risk flags, bench order). This runs the weekly pipeline up to team selection, skips news, and renders the appendix PDF.

Usage:
```bash
python -m fpl_weekly_updater appendix [--report-dir PATH] [-q|--quiet] [--log-level LEVEL]
```

---

## Command: backtest

Wraps the existing backtest analysis script to evaluate prediction quality and generate a report.

Usage:
```bash
python -m fpl_weekly_updater backtest --actual-data PATH [--gameweek-range START END] [--generate-report] [--output-dir DIR] [--season TAG] [--predictions-dir DIR] [--filter-nonplaying]
```

Required:
- `--actual-data PATH` – CSV of actual FPL results.

Optional:
- `--gameweek-range START END` – Range of gameweeks to include.
- `--generate-report` – Produce a PDF and Markdown summary (default: True).
- `--output-dir DIR` – Directory for reports (default: `reports/`).
- `--season TAG` – Season tag (default: `2024-25`).
- `--predictions-dir DIR` – Directory where predictions were archived (default: `data/backtest`).
- `--filter-nonplaying` – Exclude rows where a player did not play.

---

## Command: init-team

Builds an initial 15-player squad subject to constraints using FPL bootstrap data. Prioritizes good value (EP per price) and obeys FPL status (`a` only), budget, and per-team limits. Outputs JSON and CSV.

Usage:
```bash
python -m fpl_weekly_updater init-team [--budget 100.0] [--formation 3-5-2] [--lock "Haaland,Saka"] [--exclude "Player X"] [--max-from-team 3] [--report-dir PATH]
```

---

## Command: set-password

Stores your FPL account password securely in the OS keyring so the app never needs the secret in `.env`.

Usage:
```bash
python -m fpl_weekly_updater set-password [--email your@email]
```

If `--email` is omitted you will be prompted. The password input is hidden.

---

## Examples

- Quiet weekly run to Desktop:
```powershell
python -m fpl_weekly_updater weekly --quiet
```

- Weekly with custom output directory and no news:
```powershell
python -m fpl_weekly_updater weekly --report-dir C:\Reports\FPL --no-news --quiet
```

- Backtest with PDF report:
```powershell
python -m fpl_weekly_updater backtest --actual-data data\\actuals_2024-25.csv --gameweek-range 1 10 --generate-report
```

- Store password securely:
```powershell
python -m fpl_weekly_updater set-password --email you@example.com
```

---

## Troubleshooting

- If the console is too verbose, add `--quiet` or `--log-level WARNING`.
- To see more details, add `--verbose` or `--log-level INFO`.
- Ensure your virtual environment is active or specify the full path to `python.exe` in `.venv`.
- If Perplexity API calls are undesired or offline, use `--no-news`.
