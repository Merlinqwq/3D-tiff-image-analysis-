from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intensity_data_analysis import analyze_workbook


ROOT = Path(__file__).resolve().parents[3]
analyze_workbook(
    ROOT / "Merged_coilin_lamin_integrated_intensity.xlsx",
    Path(__file__).resolve().parent / "results" / "coilin_lamin",
    "coilin_lamin",
)
