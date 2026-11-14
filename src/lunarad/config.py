"""Single source of truth for paths and constants.

No logic and no file reading in this module; everything else imports from it.
Two environment variables relocate things without code changes (Docker, Colab):
    LUNA_DATA     folder holding the dataset and data/processed
    LUNA_RESULTS  folder for result tables and figures
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LUNA_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("LUNA_RESULTS", ROOT / "results"))

# Raw data. The double nesting of the download is defined once, here.
CT_DIR = DATA / "CT" / "CT" / "image"
VOI_IMAGE = DATA / "VOIs" / "VOIs" / "image"
VOI_MASK = DATA / "VOIs" / "VOIs" / "nodule_mask"
METADATA_XLSX = DATA / "MetadatabyAnnotation.xlsx"

# Pipeline outputs. Patient-level files stay in data/ and are never committed.
PROCESSED = DATA / "processed"
INVENTORY_CSV = PROCESSED / "image_inventory.csv"
NODULES_CSV = PROCESSED / "nodules.csv"            # modelling table: ONE label column, no ratings
LABEL_AUDIT_CSV = PROCESSED / "label_audit.csv"    # ratings and sensitivity labels, analysis only
SPLITS_CSV = PROCESSED / "splits.csv"
FEATURES_CSV = PROCESSED / "features.csv"
FAILURES_CSV = PROCESSED / "extraction_failures.csv"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# Label: malignant if the MEAN rating is >= this. Ties at exactly 3.0 count as
# malignant (decision of 2026-11-10, see docs/decisions.md).
LABEL_THRESHOLD = 3.0

# Split: one of N_OUTER outer folds is the sealed test set; the rest form N_INNER CV folds.
SEED = 42
N_OUTER = 6
N_INNER = 5

# Radiomics extraction.
BIN_WIDTH = 25
PROTOCOLS = {                      # name -> resampled voxel spacing (None = keep original)
    "original_spacing": None,
    "isotropic_1mm": (1.0, 1.0, 1.0),
}
FEATURE_PREFIX = "original_"       # PyRadiomics feature columns all start with this
N_FEATURES = 107

# Models and evaluation.
LOGREG_C = 1.0
# Decisions taken from cross-validation only, before the sealed test set is touched
# (see docs/decisions.md). (input set, model name) pairs.
FINAL_PROTOCOL = "original_spacing"
PRIMARY_MODEL = ("diameter_texture", "logreg_l2")
SECONDARY_MODEL = ("radiomics", "ensemble_avg")         # sensitivity analysis, reported alongside
# Decision threshold per model: the "recall >= 0.95" cutoff from stage 9 (cross-validation only),
# frozen before the test run. Decision of 2026-11-14, see docs/decisions.md.
FINAL_THRESHOLDS = {"primary": 0.264, "secondary": 0.266}
REFERENCE_MODEL = ("diameter", "logreg")                 # context only, never a candidate
RECALL_TARGETS = [0.90, 0.95, 0.98]                      # sensitivity the threshold must reach in CV
COST_RATIOS = [2, 5, 10, 20]                             # a missed malignancy costs k false alarms
CALIBRATION_BINS = 10
ENSEMBLE_MEMBERS = ["logreg_l2", "random_forest", "hist_gb"]   # equal-weight probability average
N_TUNING_FOLDS = 3                 # inner patient-grouped folds used to pick hyperparameters
SCANNER_COLUMNS = ["sx", "sz"]     # in-plane voxel size and slice thickness
BOOTSTRAP_REPLICATES = 2000
N_REPEATS = 10                     # repeated CV: how many different fold assignments
REPEAT_SEEDS = [SEED + i for i in range(N_REPEATS)]   # seed 42 is the locked split itself
TOP_K = 10                         # features kept per fold in the ranking-stability screen
