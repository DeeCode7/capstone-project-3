# Social Media Usage and Emotional Well-Being Classifier

End-to-end Spec-Driven ML/DL classification pipeline built on the Kaggle **Social Media Usage and Emotional Well-Being** dataset.

## Included in This Archive
* `data/train.csv`, `data/val.csv`, `data/test.csv`: Raw uploaded Kaggle dataset splits.
* `data/train_cleaned.csv`, `data/val_cleaned.csv`, `data/test_cleaned.csv`: Automatically sanitized splits with transposed `Age`/`Gender` rows repaired and corrupted lines removed.
* `models/`: Pre-trained `best_model.joblib`, `rf_pipeline.joblib`, `mlp_pipeline.joblib`, and `metrics.json` (97.09% test accuracy).
* `src/`: Modular Python package (`config.py`, `data_loader.py`, `preprocessor.py`, `train.py`, `cli.py`).
* `app.py`: Interactive Streamlit web dashboard.
* `tests/test_pipeline.py`: Automated pytest suite.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Run unit tests
pytest tests/ -v

# 2. Retrain ML & DL models
python -m src.train

# 3. Run CLI predictor
python -m src.cli --age 28 --gender Female --platform Instagram --usage 155 --posts 6 --likes 78 --comments 26 --messages 33

# 4. Launch Streamlit UI
streamlit run app.py
```
