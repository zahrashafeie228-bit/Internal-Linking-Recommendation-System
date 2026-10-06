"""
Multi-sheet Excel report generator using openpyxl.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from config.settings import EXPORTS_DIR
from src.utils.helpers import setup_logger

logger = setup_logger("excel_exporter")


class ExcelExporter:
    """Generates styled multi-tab Excel workbooks containing all analysis data."""

    @staticmethod
    def export_all(
        summary_data: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        recommendations_data: List[Dict[str, Any]],
        orphan_data: List[Dict[str, Any]],
        existing_links_data: List[Dict[str, Any]],
        evaluation_data: Optional[List[Dict[str, Any]]] = None,
        output_dir: Path = EXPORTS_DIR,
    ) -> Path:
        """
        Export all project datasets to a multi-sheet Excel (.xlsx) file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filepath = output_dir / f"internal_link_analysis_{timestamp}.xlsx"

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # 1. Summary Sheet
            df_summary = pd.DataFrame([summary_data])
            df_summary.to_excel(writer, sheet_name="اطلاعات کلی و خلاصه", index=False)

            # 2. Pages Sheet
            if pages_data:
                df_pages = pd.DataFrame(pages_data)
                df_pages.to_excel(writer, sheet_name="صفحات خزش‌شده", index=False)

            # 3. Recommendations Sheet
            if recommendations_data:
                df_recs = pd.DataFrame(recommendations_data)
                df_recs.to_excel(writer, sheet_name="پیشنهادهای لینک داخلی", index=False)

            # 4. Orphan Pages Sheet
            if orphan_data:
                df_orphans = pd.DataFrame(orphan_data)
                df_orphans.to_excel(writer, sheet_name="صفحات یتیم و ضعیف", index=False)

            # 5. Existing Links Sheet
            if existing_links_data:
                df_links = pd.DataFrame(existing_links_data)
                df_links.to_excel(writer, sheet_name="لینک‌های داخلی موجود", index=False)

            # 6. Evaluation Sheet
            if evaluation_data:
                df_evals = pd.DataFrame(evaluation_data)
                df_evals.to_excel(writer, sheet_name="ارزیابی‌های دستی", index=False)

        logger.info(f"Exported multi-sheet Excel report to {filepath}")
        return filepath
