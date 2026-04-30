from __future__ import annotations

from pathlib import Path

# Centralized project paths (edit only here).
PROJECT_ROOT = Path(r"D:\SPDC_Project")

# Data
ECMBI_TOMOGRAPHY_DIR = PROJECT_ROOT / "03_Data" / "ECMBI_Tomography"

# Outputs
THESIS_FIG_DIR = PROJECT_ROOT / "01_Thesis_Figures"
RESULTS_CSV_DIR = PROJECT_ROOT / "05_Results" / "CSV_Data"
RESULTS_GRAPHS_DIR = PROJECT_ROOT / "05_Results" / "Graphs"
RESULTS_PDF_DIR = PROJECT_ROOT / "05_Results" / "PDF_Reports"

# Optional datasets
PHOTON_TRACE_CSV = PROJECT_ROOT / "03_Data" / "priyansh50us.csv"

# Crystal calibration parameters for BBO
# Update these values from a trusted SNLO measurement or lab characterization.
BBO_N_O_583 = 1.670120      # Ordinary index at 583 nm
BBO_N_E_583 = 1.551929      # Extraordinary index at 583 nm
BBO_N_O_900 = 1.657         # Ordinary index at 900 nm
BBO_N_E_900 = 1.541         # Extraordinary index at 900 nm
BBO_CRYSTAL_THICKNESS_MM = 1.0

# Uncertainty estimates for error propagation
BBO_N_O_583_UNC = 1e-6
BBO_N_E_583_UNC = 1e-6
BBO_N_O_900_UNC = 1e-6
BBO_N_E_900_UNC = 1e-6
BBO_CRYSTAL_THICKNESS_UNC_MM = 0.01

# Compensator properties
COMPENSATOR_MATERIAL = "Quartz"
COMPENSATOR_BIREFRINGENCE = 0.009
COMPENSATOR_BIREFRINGENCE_UNC = 0.0001


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
