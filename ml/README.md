## ML

Machine learning workspace for:
- data preparation
- feature engineering
- training and evaluation
- artifact export for backend serving

Code root: `ml/src/`

### Quick Start

1. Install dependencies:
   - `pip install -r ml/requirements.txt`
2. Create or reuse deterministic train/val/test split:
   - `python -m ml.src.data.create_split`
   - or monthly ratio split: `python -m ml.src.data.create_split --strategy monthly_chronological_ratio --force-rebuild`
3. Train all models (validation-driven model selection):
   - `python -m ml.src.models.train_pipeline`
4. Evaluate on held-out test split:
   - `python -m ml.src.evaluation.evaluate_all_test`
5. Compare both split scenarios for checkpoint discussion:
   - `python -m ml.src.evaluation.compare_split_strategies`
6. Check outputs:
   - `ml/artifacts/*_model.joblib`
   - `ml/artifacts/*_training_report.json`
   - `docs/metrics_validation.md`
   - `docs/evaluation/*.md`
   - `docs/checkpoint_split_comparison.md`

### Activity Pipelines

1. Create activity split:
   - `python -m ml.src.data.create_activity_split --strategy monthly_chronological_ratio --force-rebuild`
2. Train activity models:
   - `python -m ml.src.models.train_activity_pipeline --strategy monthly_chronological_ratio --force-rebuild-split`
3. Evaluate on activity test split:
   - `python -m ml.src.evaluation.evaluate_activity_all_test --strategy monthly_chronological_ratio`

### Daily Summary Pipelines

1. Create daily split:
   - `python -m ml.src.data.create_daily_split --strategy monthly_chronological_ratio --force-rebuild`
2. Train daily models:
   - `python -m ml.src.models.train_daily_pipeline --strategy monthly_chronological_ratio --force-rebuild-split`
3. Evaluate on daily test split:
   - `python -m ml.src.evaluation.evaluate_daily_all_test --strategy monthly_chronological_ratio`
