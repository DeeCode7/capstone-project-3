"""Data ingestion, defensive anomaly cleaning, and deterministic synthetic fallback."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES,
    DATA_DIR,
    ID_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    SYNTHETIC_SAMPLES,
    TARGET_COLUMN,
    TEST_SIZE,
    VALID_EMOTIONS,
    VALID_GENDERS,
    VALID_PLATFORMS,
)

logger = logging.getLogger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize raw dataframe according to Phase 1 defensive data cleaning rules."""
    cleaned = df.copy()

    # 1. Strip column whitespace and drop completely empty rows
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.dropna(how="all").reset_index(drop=True)

    # 2. Remove embedded duplicate header rows
    if "Age" in cleaned.columns:
        cleaned = cleaned[cleaned["Age"].astype(str).str.strip() != "Age"].copy()

    # 3. Drop identifier column if present
    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COLUMN])

    # 4. Recover transposed Age and Gender values
    if "Age" in cleaned.columns and "Gender" in cleaned.columns:
        age_str = cleaned["Age"].astype(str).str.strip()
        gender_str = cleaned["Gender"].astype(str).str.strip()

        shifted_mask = age_str.isin(VALID_GENDERS) & pd.to_numeric(
            gender_str, errors="coerce"
        ).notna()
        if shifted_mask.any():
            logger.info(
                "Recovering %d rows with transposed Age and Gender values.",
                int(shifted_mask.sum()),
            )
            cleaned.loc[shifted_mask, ["Age", "Gender"]] = cleaned.loc[
                shifted_mask, ["Gender", "Age"]
            ].values

    # 5. Clean categorical columns
    if "Gender" in cleaned.columns:
        cleaned["Gender"] = cleaned["Gender"].astype(str).str.strip()
        cleaned.loc[~cleaned["Gender"].isin(VALID_GENDERS), "Gender"] = np.nan

    if "Platform" in cleaned.columns:
        cleaned["Platform"] = cleaned["Platform"].astype(str).str.strip().str.title()
        cleaned["Platform"] = cleaned["Platform"].replace({"Linkedin": "LinkedIn"})
        cleaned.loc[~cleaned["Platform"].isin(VALID_PLATFORMS), "Platform"] = np.nan

    # 6. Safe numeric coercion and domain bounding
    bounds: dict[str, tuple[float, float]] = {
        "Age": (15.0, 70.0),
        "Daily_Usage_Time (minutes)": (0.0, 1440.0),
        "Posts_Per_Day": (0.0, 100.0),
        "Likes_Received_Per_Day": (0.0, 5000.0),
        "Comments_Received_Per_Day": (0.0, 1000.0),
        "Messages_Sent_Per_Day": (0.0, 1000.0),
    }
    for col in NUMERIC_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            low, high = bounds[col]
            cleaned[col] = cleaned[col].clip(lower=low, upper=high)

    # 7. Target sanitization
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = (
            cleaned[TARGET_COLUMN].astype(str).str.strip().str.title()
        )
        cleaned = cleaned[cleaned[TARGET_COLUMN].isin(VALID_EMOTIONS)].copy()

    return cleaned.reset_index(drop=True)


def generate_synthetic_dataset(
    n_samples: int = SYNTHETIC_SAMPLES, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate a deterministic synthetic dataset mirroring Kaggle distributions."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    emotion_profiles: dict[str, dict[str, object]] = {
        "Happiness": {
            "platforms": ["Instagram", "Snapchat", "Whatsapp"],
            "usage": (150, 25),
            "posts": (5.7, 1.4),
            "likes": (76, 18),
            "comments": (27, 5),
            "messages": (33, 7),
            "age": (28, 4),
        },
        "Sadness": {
            "platforms": ["Instagram", "Facebook", "Snapchat"],
            "usage": (84, 25),
            "posts": (3.1, 1.5),
            "likes": (33, 18),
            "comments": (13, 5),
            "messages": (21, 7),
            "age": (27, 4),
        },
        "Anger": {
            "platforms": ["Twitter", "Facebook"],
            "usage": (91, 15),
            "posts": (3.7, 1.1),
            "likes": (39, 12),
            "comments": (19, 6),
            "messages": (23, 4),
            "age": (28, 3),
        },
        "Anxiety": {
            "platforms": ["Instagram", "Twitter", "Snapchat"],
            "usage": (95, 35),
            "posts": (3.0, 1.7),
            "likes": (40, 22),
            "comments": (15, 8),
            "messages": (24, 7),
            "age": (28, 3),
        },
        "Boredom": {
            "platforms": ["Facebook", "LinkedIn", "Telegram"],
            "usage": (65, 15),
            "posts": (1.6, 1.0),
            "likes": (16, 7),
            "comments": (7, 3.5),
            "messages": (14, 4),
            "age": (30, 4),
        },
        "Neutral": {
            "platforms": ["LinkedIn", "Whatsapp", "Telegram"],
            "usage": (78, 24),
            "posts": (2.4, 1.2),
            "likes": (27, 16),
            "comments": (11, 5),
            "messages": (18, 6),
            "age": (25, 4),
        },
    }

    for idx in range(1, n_samples + 1):
        emotion = VALID_EMOTIONS[idx % len(VALID_EMOTIONS)]
        profile = emotion_profiles[emotion]

        platform = str(rng.choice(profile["platforms"]))
        gender = str(rng.choice(VALID_GENDERS, p=[0.46, 0.44, 0.10]))

        age_mean, age_std = profile["age"]
        u_mean, u_std = profile["usage"]
        p_mean, p_std = profile["posts"]
        l_mean, l_std = profile["likes"]
        c_mean, c_std = profile["comments"]
        m_mean, m_std = profile["messages"]

        records.append(
            {
                ID_COLUMN: idx,
                "Age": round(float(np.clip(rng.normal(age_mean, age_std), 18, 60)), 0),
                "Gender": gender,
                "Platform": platform,
                "Daily_Usage_Time (minutes)": round(
                    float(np.clip(rng.normal(u_mean, u_std), 10, 480)), 0
                ),
                "Posts_Per_Day": round(
                    float(np.clip(rng.normal(p_mean, p_std), 0, 25)), 0
                ),
                "Likes_Received_Per_Day": round(
                    float(np.clip(rng.normal(l_mean, l_std), 0, 500)), 0
                ),
                "Comments_Received_Per_Day": round(
                    float(np.clip(rng.normal(c_mean, c_std), 0, 200)), 0
                ),
                "Messages_Sent_Per_Day": round(
                    float(np.clip(rng.normal(m_mean, m_std), 0, 200)), 0
                ),
                TARGET_COLUMN: emotion,
            }
        )

    df = pd.DataFrame(records)
    df.loc[0, ["Age", "Gender"]] = [df.loc[0, "Gender"], df.loc[0, "Age"]]
    df.loc[1, "Likes_Received_Per_Day"] = np.nan
    df.loc[2, "Platform"] = np.nan

    return df


def load_datasets(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load Kaggle CSV splits if available, otherwise use deterministic synthetic fallback."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    val_path = data_dir / "val.csv"

    if train_path.exists():
        logger.info("Found local Kaggle dataset at %s", train_path)
        train_df = clean_dataframe(pd.read_csv(train_path, on_bad_lines="skip"))
        train_df.to_csv(data_dir / "train_cleaned.csv", index=False)

        if val_path.exists():
            val_df = clean_dataframe(pd.read_csv(val_path, on_bad_lines="skip"))
            val_df.to_csv(data_dir / "val_cleaned.csv", index=False)
            train_df = pd.concat([train_df, val_df], ignore_index=True)

        if test_path.exists():
            test_df = clean_dataframe(pd.read_csv(test_path, on_bad_lines="skip"))
            test_df.to_csv(data_dir / "test_cleaned.csv", index=False)
        else:
            train_df, test_df = train_test_split(
                train_df,
                test_size=TEST_SIZE,
                stratify=train_df[TARGET_COLUMN],
                random_state=RANDOM_SEED,
            )
        source_type = "kaggle_csv"
    else:
        logger.warning(
            "Local Kaggle train.csv not found in %s. Generating deterministic synthetic dataset.",
            data_dir,
        )
        raw_synthetic = generate_synthetic_dataset()
        synthetic_path = data_dir / "synthetic_social_media.csv"
        raw_synthetic.to_csv(synthetic_path, index=False)

        cleaned_synthetic = clean_dataframe(raw_synthetic)
        train_df, test_df = train_test_split(
            cleaned_synthetic,
            test_size=TEST_SIZE,
            stratify=cleaned_synthetic[TARGET_COLUMN],
            random_state=RANDOM_SEED,
        )
        source_type = "synthetic_fallback"

    return (
        train_df[ALL_FEATURES + [TARGET_COLUMN]].reset_index(drop=True),
        test_df[ALL_FEATURES + [TARGET_COLUMN]].reset_index(drop=True),
        source_type,
    )
