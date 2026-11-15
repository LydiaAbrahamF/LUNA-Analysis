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
PROCESSED = Path(os.environ.get("LUNA_PROCESSED", DATA / "processed"))   # derived tables; relocatable so that
# a second run (for example in Docker) can read the same raw data without overwriting these files
INVENTORY_CSV = PROCESSED / "image_inventory.csv"
NODULES_CSV = PROCESSED / "nodules.csv"            # modelling table: ONE label column, no ratings
LABEL_AUDIT_CSV = PROCESSED / "label_audit.csv"    # ratings and sensitivity labels, analysis only
SPLITS_CSV = PROCESSED / "splits.csv"
FEATURES_CSV = PROCESSED / "features.csv"
FAILURES_CSV = PROCESSED / "extraction_failures.csv"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# Label rule. Default = the LUNA16 / LIDC dataset authors' rule: a nodule is malignant if AT LEAST
# TWO radiologists rated it above 3 (a 4 or a 5).
# Other rules exist for sensitivity analyses only. Choose with the environment variable
# LUNA_LABEL_RULE: dataset | mean_ge3 | mean_gt3 | majority. If you change the rule, also point
# LUNA_DATA and LUNA_RESULTS at separate folders so the two analyses never mix.
LABEL_RULE = os.environ.get("LUNA_LABEL_RULE", "dataset")
LABEL_THRESHOLD = 3.0              # used by the mean-based rules only
N_HIGH_REQUIRED = 2                # dataset rule: raters needed above 3

# Split: one of N_OUTER outer folds is the sealed test set; the rest form N_INNER CV folds.
SEED = 42
N_OUTER = 6
N_INNER = 5

# Radiomics extraction.
BIN_WIDTH = 25
# Interpolator for the resampled protocol. LINEAR, not B-spline: 37 of the 994 nodule crops have only
# 2 slices, and a cubic B-spline needs more samples than that along an axis. On those crops B-spline
# resampling returned different (sometimes absurd) intensities on every run, which also made the
# compiled PyRadiomics code fail or crash. Linear interpolation is defined for 2 slices and repeatable.
INTERPOLATOR = "sitkLinear"
PROTOCOLS = {                      # name -> resampled voxel spacing (None = keep original)
    "original_spacing": None,
    "isotropic_1mm": (1.0, 1.0, 1.0),
}
FEATURE_PREFIX = "original_"       # PyRadiomics feature columns all start with this
N_FEATURES = 107

# Models and evaluation.
LOGREG_C = 1.0
# Decisions taken from cross-validation only, before the sealed test set is touched
# (input set, model name) pairs.
FINAL_PROTOCOL = "original_spacing"
PRIMARY_MODEL = ("radiomics", "logreg_l2")
SECONDARY_MODEL = ("radiomics", "ensemble_avg")         # sensitivity analysis, reported alongside
# Decision threshold per model: the "recall >= 0.95" cutoff from stage 9 (cross-validation only),
# frozen before the test run (dataset label rule, stage 9 of 2026-11-14). If the label rule or the
# models change, set both to None, rerun stage 9, copy its cutoffs here and commit BEFORE the test run.
FINAL_THRESHOLDS = {"primary": 0.154, "secondary": 0.129}
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
