from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import importlib.util

# Import pipelines and helpers
# NOTE: Avoid importing heavy modules at import time; keep appendix lightweight.

def _load_backtest_main() -> Optional[callable]:
    """Dynamically load the backtest script's main() function by file path.

    Supports running from source tree and from a PyInstaller one-file EXE (using sys._MEIPASS).
    """
    candidates: list[Path] = []
    # Project-relative scripts
    candidates.append(Path.cwd() / "scripts" / "run_backtest_analysis.py")
    # When executed as a module from project root
    candidates.append(Path(__file__).resolve().parents[1] / "scripts" / "run_backtest_analysis.py")
    # PyInstaller one-file extraction dir
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "scripts" / "run_backtest_analysis.py")
    # PyInstaller one-dir: data files live next to the executable
    try:
        exe_dir = Path(sys.executable).parent
        candidates.append(exe_dir / "scripts" / "run_backtest_analysis.py")
    except Exception:
        pass
    
    for path in candidates:
        try:
            if path and path.exists():
                spec = importlib.util.spec_from_file_location("run_backtest_analysis", str(path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                    if hasattr(module, "main"):
                        return getattr(module, "main")
        except Exception:
            continue
    return None

try:
    from scripts.set_fpl_password import prompt_and_store as set_password  # type: ignore
except Exception:  # pragma: no cover
    set_password = None


def _apply_log_level(args: argparse.Namespace) -> None:
    """Apply log level from CLI flags without reconfiguring handlers.

    Respects existing basicConfig set up by modules. Only adjusts the root level.
    """
    level: Optional[int] = None
    if getattr(args, "log_level", None):
        level = getattr(logging, str(args.log_level).upper(), None)
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    elif getattr(args, "verbose", False):
        level = logging.INFO

    if level is not None:
        logging.getLogger().setLevel(level)


def _add_common_verbosity_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--log-level", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help="Set log level")
    p.add_argument("-q", "--quiet", action="store_true", help="Alias for --log-level WARNING")
    p.add_argument("-v", "--verbose", action="store_true", help="Alias for --log-level INFO")


def _cmd_weekly(args: argparse.Namespace) -> int:
    _apply_log_level(args)
    # Lazy import to avoid pulling ML/xgboost when not needed
    from fpl_weekly_updater.main import run_weekly_update  # type: ignore
    # Lazy load settings only for weekly path
    from fpl_weekly_updater.config.settings import load_settings  # type: ignore
    kwargs: dict = {}
    if args.gameweek:
        kwargs["generate_predictions_for_gw"] = args.gameweek
    if args.no_news:
        kwargs["skip_news"] = True
    if args.report_dir:
        kwargs["report_dir"] = Path(args.report_dir)
    if args.appendix:
        kwargs["appendix"] = True

    path = run_weekly_update(**kwargs)
    if path is None:
        print("Weekly update failed.")
        return 1
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    _apply_log_level(args)
    backtest_main = _load_backtest_main()
    if backtest_main is None:
        print("Backtest tooling not available in this build.")
        return 1
    # Re-map CLI to the existing backtest script interface
    cli = ["--actual-data", args.actual_data]
    if args.gameweek_range:
        cli += ["--gameweek-range", str(args.gameweek_range[0]), str(args.gameweek_range[1])]
    if args.generate_report:
        cli += ["--generate-report"]
    if args.output_dir:
        cli += ["--output-dir", args.output_dir]
    if args.season:
        cli += ["--season", args.season]
    if args.predictions_dir:
        cli += ["--predictions-dir", args.predictions_dir]
    if args.filter_nonplaying:
        cli += ["--filter-nonplaying"]

    sys.argv = ["fpl backtest"] + cli
    return int(backtest_main())


def _cmd_set_password(args: argparse.Namespace) -> int:
    _apply_log_level(args)
    if set_password is None:
        print("Keyring password helper not available in this build.")
        return 1
    set_password(args.email)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpl", description="FPL 2025 Toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    # weekly
    p_weekly = sub.add_parser("weekly", help="Run the main weekly update and generate the report")
    _add_common_verbosity_flags(p_weekly)
    p_weekly.add_argument("--gameweek", type=int, help="Target gameweek (for backtesting)")
    p_weekly.add_argument("--report-dir", type=str, help="Directory to save output report(s)")
    p_weekly.add_argument("--appendix", action="store_true", help="Also generate the appendix variant")
    p_weekly.add_argument("--no-news", action="store_true", help="Skip Perplexity/news analysis")
    p_weekly.set_defaults(func=_cmd_weekly)

    # appendix (redefined): Generate the ML Pipeline Report by wrapping backtest analysis
    p_appendix = sub.add_parser("appendix", help="Generate the ML Pipeline (backtest) report PDF")
    _add_common_verbosity_flags(p_appendix)
    p_appendix.add_argument("--actual-data", type=str, help="Path to CSV with actual FPL results (defaults to data/actuals_2024-25.csv if present)")
    p_appendix.add_argument("--output-dir", type=str, help="Directory to save report (default: reports)")
    p_appendix.add_argument("--predictions-dir", type=str, help="Directory containing archived predictions (default: data/backtest)")
    def _cmd_appendix(args: argparse.Namespace) -> int:
        # Force clean console output by default for appendix runs
        logging.getLogger().setLevel(logging.ERROR)
        # Determine actuals path
        actual_path = args.actual_data
        default_actual = Path("data/actuals_2024-25.csv")
        if not actual_path:
            if default_actual.exists():
                actual_path = str(default_actual)
        if not actual_path:
            print("Appendix requires --actual-data or a default 'data/actuals_2024-25.csv' present.")
            return 2
        # Try the full backtest analysis script first
        try:
            backtest_main = _load_backtest_main()
            if backtest_main is not None:
                cli = ["--actual-data", actual_path, "--generate-report"]
                if args.output_dir:
                    cli += ["--output-dir", args.output_dir]
                if args.predictions_dir:
                    cli += ["--predictions-dir", args.predictions_dir]
                # Use defaults for season/predictions-dir; users can use 'backtest' for advanced control
                sys.argv = ["fpl appendix"] + cli
                return int(backtest_main())
        except Exception as e:
            # Fall through to lite
            pass
        # Fallback: Lightweight backtest that only depends on pandas/numpy/matplotlib
        try:
            from fpl_weekly_updater.analysis.backtest_lite import run_backtest_lite  # type: ignore
            out_dir = Path(args.output_dir) if args.output_dir else Path("reports")
            pred_dir = Path(args.predictions_dir) if args.predictions_dir else Path("data/backtest")
            res = run_backtest_lite(Path(actual_path), pred_dir, out_dir)
            if res.pdf_path or res.md_path:
                print("Backtest (lite) completed.")
                return 0
            print("Backtest (lite) did not produce output.")
            return 1
        except Exception as e:
            print(f"Backtest tooling not available: {e}")
            return 1
    p_appendix.set_defaults(func=_cmd_appendix)

    # backtest (wrap existing script)
    p_back = sub.add_parser("backtest", help="Run historical backtest analysis and optional report")
    _add_common_verbosity_flags(p_back)
    p_back.add_argument("--actual-data", required=True, help="Path to CSV with actual FPL results")
    p_back.add_argument("--gameweek-range", nargs=2, type=int, metavar=("START", "END"), help="Gameweek range")
    p_back.add_argument("--generate-report", action="store_true", default=True, help="Generate PDF report")
    p_back.add_argument("--output-dir", default="reports", help="Output directory (default: reports)")
    p_back.add_argument("--season", default="2024-25", help="Season tag (default: 2024-25)")
    p_back.add_argument("--predictions-dir", default="data/backtest", help="Directory of predictions")
    p_back.add_argument("--filter-nonplaying", action="store_true", default=False, help="Exclude non-playing rows from actuals")
    p_back.set_defaults(func=_cmd_backtest)

    # init-team (implemented baseline)
    p_init = sub.add_parser("init-team", help="Build an initial 15-player squad within constraints")
    _add_common_verbosity_flags(p_init)
    p_init.add_argument("--budget", type=float, default=100.0, help="Budget in millions (default: 100.0)")
    p_init.add_argument("--formation", type=str, default="3-5-2", help="Starting XI formation for info (e.g., 3-5-2)")
    p_init.add_argument("--lock", type=str, default="", help="Comma-separated names to lock into the squad")
    p_init.add_argument("--exclude", type=str, default="", help="Comma-separated names to exclude")
    p_init.add_argument("--max-from-team", type=int, default=3, help="Max players per Premier League team (default: 3)")
    p_init.add_argument("--report-dir", type=str, help="Directory to save initial squad files (JSON/CSV)")
    def _cmd_init(args: argparse.Namespace) -> int:
        _apply_log_level(args)
        # Lazy import solver
        from fpl_weekly_updater.team_selector.solver import select_initial_squad  # type: ignore
        # Lazy import client to avoid pulling requests for appendix/backtest
        from fpl_weekly_updater.apis.fpl_client import FPLClient  # type: ignore
        from fpl_weekly_updater.config.settings import load_settings  # type: ignore
        settings = load_settings()
        client = FPLClient(
            base_url=settings.api.base_url,
            timeout=settings.api.timeout,
            rate_limit_delay=settings.api.rate_limit_delay,
            retry_attempts=settings.api.retry_attempts,
            session_cookie=settings.fpl_session_cookie,
        )
        bootstrap = client.get_bootstrap()
        locks = [s.strip() for s in (args.lock or "").split(",") if s.strip()]
        excludes = [s.strip() for s in (args.exclude or "").split(",") if s.strip()]
        result = select_initial_squad(
            bootstrap=bootstrap,
            budget_millions=float(args.budget),
            formation=args.formation,
            lock_names=locks,
            exclude_names=excludes,
            max_from_team=int(args.max_from_team),
        )
        out_dir = Path(args.report_dir) if args.report_dir else Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"initial_squad_{ts}.json"
        csv_path = out_dir / f"initial_squad_{ts}.csv"
        import json as _json, csv as _csv
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(result, f, indent=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["id","name","position","team","price","ownership"])
            writer.writeheader()
            for row in result.get("squad", []):
                writer.writerow(row)
        logging.getLogger(__name__).info("Initial squad saved: %s; %s", json_path, csv_path)
        return 0
    p_init.set_defaults(func=_cmd_init)

    # set-password
    p_pw = sub.add_parser("set-password", help="Store FPL password in the OS keyring")
    _add_common_verbosity_flags(p_pw)
    p_pw.add_argument("--email", type=str, help="FPL account email (will prompt if omitted)")
    p_pw.set_defaults(func=_cmd_set_password)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    if argv is not None:
        sys.argv = ["fpl"] + list(argv)
    parser = build_parser()
    args = parser.parse_args()

    # Ensure default log level from env if configured
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        try:
            logging.getLogger().setLevel(getattr(logging, env_level.upper()))
        except Exception:
            pass

    fn = getattr(args, "func", None)
    if not fn:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
