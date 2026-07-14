from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "docs" / "evaluation"

HOURLY_DATA_PATH = RAW_DATA_DIR / "hourly_observations.csv"
ACTIVITY_DATA_PATH = RAW_DATA_DIR / "activity_recommendations.csv"
DAILY_DATA_PATH = RAW_DATA_DIR / "daily_summaries.csv"
SPLIT_ASSIGNMENTS_PATH = SPLITS_DIR / "hourly_observations_split.csv"
SPLIT_MANIFEST_PATH = ARTIFACTS_DIR / "split_manifest.json"
ACTIVITY_SPLIT_ASSIGNMENTS_PATH = SPLITS_DIR / "activity_recommendations_split.csv"
ACTIVITY_SPLIT_MANIFEST_PATH = ARTIFACTS_DIR / "activity_split_manifest.json"
DAILY_SPLIT_ASSIGNMENTS_PATH = SPLITS_DIR / "daily_summaries_split.csv"
DAILY_SPLIT_MANIFEST_PATH = ARTIFACTS_DIR / "daily_split_manifest.json"

UMBRELLA_MODEL_PATH = ARTIFACTS_DIR / "umbrella_model.joblib"
CLOTHING_MODEL_PATH = ARTIFACTS_DIR / "clothing_model.joblib"
SUITABILITY_MODEL_PATH = ARTIFACTS_DIR / "suitability_model.joblib"
ACTIVITY_GO_NO_MODEL_PATH = ARTIFACTS_DIR / "activity_go_no_model.joblib"
ACTIVITY_SUITABILITY_MODEL_PATH = ARTIFACTS_DIR / "activity_suitability_model.joblib"
DAILY_UMBRELLA_MODEL_PATH = ARTIFACTS_DIR / "daily_umbrella_model.joblib"
DAILY_BEST_HOUR_MODEL_PATH = ARTIFACTS_DIR / "daily_best_hour_model.joblib"
DAILY_CLOTHING_MORNING_MODEL_PATH = ARTIFACTS_DIR / "daily_clothing_morning_model.joblib"
DAILY_CLOTHING_AFTERNOON_MODEL_PATH = ARTIFACTS_DIR / "daily_clothing_afternoon_model.joblib"
DAILY_CLOTHING_EVENING_MODEL_PATH = ARTIFACTS_DIR / "daily_clothing_evening_model.joblib"
DAILY_SUITABILITY_MODEL_PATH = ARTIFACTS_DIR / "daily_suitability_model.joblib"

UMBRELLA_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "umbrella_training_report.json"
CLOTHING_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "clothing_training_report.json"
SUITABILITY_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "suitability_training_report.json"
ACTIVITY_GO_NO_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "activity_go_no_training_report.json"
ACTIVITY_SUITABILITY_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "activity_suitability_training_report.json"
DAILY_UMBRELLA_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "daily_umbrella_training_report.json"
DAILY_BEST_HOUR_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "daily_best_hour_training_report.json"
DAILY_CLOTHING_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "daily_clothing_training_report.json"
DAILY_SUITABILITY_TRAINING_REPORT_PATH = ARTIFACTS_DIR / "daily_suitability_training_report.json"

UMBRELLA_TEST_REPORT_PATH = REPORTS_DIR / "umbrella_test_metrics.md"
CLOTHING_TEST_REPORT_PATH = REPORTS_DIR / "clothing_test_metrics.md"
SUITABILITY_TEST_REPORT_PATH = REPORTS_DIR / "suitability_test_metrics.md"
ALL_TEST_REPORT_PATH = REPORTS_DIR / "all_test_metrics.md"
ACTIVITY_GO_NO_TEST_REPORT_PATH = REPORTS_DIR / "activity_go_no_test_metrics.md"
ACTIVITY_SUITABILITY_TEST_REPORT_PATH = REPORTS_DIR / "activity_suitability_test_metrics.md"
ACTIVITY_ALL_TEST_REPORT_PATH = REPORTS_DIR / "activity_all_test_metrics.md"
DAILY_UMBRELLA_TEST_REPORT_PATH = REPORTS_DIR / "daily_umbrella_test_metrics.md"
DAILY_BEST_HOUR_TEST_REPORT_PATH = REPORTS_DIR / "daily_best_hour_test_metrics.md"
DAILY_CLOTHING_TEST_REPORT_PATH = REPORTS_DIR / "daily_clothing_test_metrics.md"
DAILY_SUITABILITY_TEST_REPORT_PATH = REPORTS_DIR / "daily_suitability_test_metrics.md"
DAILY_ALL_TEST_REPORT_PATH = REPORTS_DIR / "daily_all_test_metrics.md"
