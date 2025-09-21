import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fpl_weekly_update.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from models.ml_predictor import MLPredictor
from scripts.run_backtest_analysis import generate_pdf_report, generate_mock_actual_results, load_actual_results

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to run the weekly backtest and generate a report."""
    logger.info("Starting weekly backtest analysis...")
    config = load_config()
    predictor = MLPredictor(config)

    # Generate mock actual results for the purpose of this automated report
    backtest_dir = Path('data/backtest')
    actual_results = None
    if backtest_dir.exists():
        mock_file = generate_mock_actual_results(backtest_dir)
        if mock_file:
            actual_results = load_actual_results(mock_file)
            logger.info(f"Using generated mock actual results: {mock_file}")

    if actual_results is None:
        logger.error("Error: No actual results could be generated. Aborting.")
        return

    # Run the backtest
    backtest_results = predictor.run_historical_backtest(actual_results)

    if not backtest_results:
        logger.warning("Backtest analysis failed to produce results.")
        return

    # Generate the PDF report on the desktop
    desktop_path = Path.home() / 'Desktop'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = desktop_path / f"weekly_backtest_report_{timestamp}.pdf"

    logger.info(f"Generating backtest report at: {report_path}")
    generate_pdf_report(backtest_results, report_path)

    logger.info("Weekly backtest analysis complete.")

if __name__ == "__main__":
    main()
