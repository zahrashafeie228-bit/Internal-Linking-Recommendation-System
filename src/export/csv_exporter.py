"""
CSV exporter with UTF-8 BOM encoding for Persian text compatibility in Excel.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from config.settings import EXPORTS_DIR
from src.utils.helpers import setup_logger

logger = setup_logger("csv_exporter")


class CSVExporter:
    """Exports dataframes and dictionary lists to Persian UTF-8-SIG CSV files."""

    @staticmethod
    def export_to_csv(
        data: List[Dict[str, Any]] or pd.DataFrame,
        filename: str,
        output_dir: Path = EXPORTS_DIR,
    ) -> Path:
        """
        Export list of dictionaries or DataFrame to UTF-8-SIG CSV.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        # Use utf-8-sig to preserve Persian characters in Windows Excel
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"Exported CSV to {filepath}")
        return filepath
