"""Unit tests for data cleaning, preprocessing, training, and CLI inference."""

import pandas as pd

from src.cli import predict_single_profile
from src.config import ALL_FEATURES, QUALITY_GATE_MACRO_F1, TARGET_COLUMN
from src.data_loader import clean_dataframe, generate_synthetic_dataset
from src.preprocessor import build_preprocessor
from src.train import train_and_evaluate


def test_defensive_cleaner_recovers_transposed_and_corrupted_rows() -> None:
    """Verify transposed Age/Gender recovery and invalid row removal."""
    raw_df = pd.DataFrame(
        [
            {
                "User_ID": 1,
                "Age": "Female",
                "Gender": "27",
                "Platform": "Instagram",
                "Daily_Usage_Time (minutes)": 165,
                "Posts_Per_Day": 6,
                "Likes_Received_Per_Day": 85,
                "Comments_Received_Per_Day": 30,
                "Messages_Sent_Per_Day": 28,
                "Dominant_Emotion": "Happiness",
            },
            {
                "User_ID": "Tabii",
                "Age": "iste mevcut veri kumesini",
                "Gender": None,
                "Platform": None,
                "Daily_Usage_Time (minutes)": None,
                "Posts_Per_Day": None,
                "Likes_Received_Per_Day": None,
                "Comments_Received_Per_Day": None,
                "Messages_Sent_Per_Day": None,
                "Dominant_Emotion": None,
            },
            {
                "User_ID": 3,
                "Age": "24",
                "Gender": "Female",
                "Platform": "Instagram",
                "Daily_Usage_Time (minutes)": 140,
                "Posts_Per_Day": 6,
                "Likes_Received_Per_Day": 50,
                "Comments_Received_Per_Day": 21,
                "Messages_Sent_Per_Day": 29,
                "Dominant_Emotion": "Agression",
            },
        ]
    )

    cleaned = clean_dataframe(raw_df)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "Age"] == 27.0
    assert cleaned.loc[0, "Gender"] == "Female"
    assert cleaned.loc[0, TARGET_COLUMN] == "Happiness"


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    """Ensure ColumnTransformer imputes missing values and encodes without error."""
    df = clean_dataframe(generate_synthetic_dataset(n_samples=60))
    X = df[ALL_FEATURES].copy()
    X.loc[0, "Age"] = None
    X.loc[1, "Gender"] = None

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not pd.isna(transformed).any()


def test_training_and_single_profile_inference() -> None:
    """Verify training meets the Macro-F1 quality gate and CLI inference succeeds."""
    results = train_and_evaluate()
    assert results["best_macro_f1"] >= QUALITY_GATE_MACRO_F1

    sample_profile = {
        "Age": 28.0,
        "Gender": "Female",
        "Platform": "Instagram",
        "Daily_Usage_Time (minutes)": 155.0,
        "Posts_Per_Day": 6.0,
        "Likes_Received_Per_Day": 78.0,
        "Comments_Received_Per_Day": 26.0,
        "Messages_Sent_Per_Day": 33.0,
    }
    pred = predict_single_profile(sample_profile, model_choice="best")
    assert pred["predicted_emotion"] in [
        "Happiness",
        "Sadness",
        "Anger",
        "Anxiety",
        "Boredom",
        "Neutral",
    ]
    assert abs(sum(pred["confidence_scores"].values()) - 100.0) < 0.5
