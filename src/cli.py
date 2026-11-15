"""Command-Line Interface for predicting Dominant Emotion from social media usage."""

import argparse
import sys
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    MLP_MODEL_PATH,
    RF_MODEL_PATH,
    VALID_GENDERS,
    VALID_PLATFORMS,
)
from src.train import train_and_evaluate


def load_model_bundle(model_choice: str = "best") -> dict[str, Any]:
    """Load the serialized model pipeline and label encoder, training if missing."""
    path_map: dict[str, Path] = {
        "best": BEST_MODEL_PATH,
        "rf": RF_MODEL_PATH,
        "mlp": MLP_MODEL_PATH,
    }
    target_path = path_map.get(model_choice, BEST_MODEL_PATH)

    if not target_path.exists():
        print("Model artifacts not found. Running initial training pipeline...")
        train_and_evaluate()

    return joblib.load(target_path)


def predict_single_profile(
    profile: dict[str, Any], model_choice: str = "best"
) -> dict[str, Any]:
    """Run inference on a single user feature dictionary."""
    bundle = load_model_bundle(model_choice)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]

    input_df = pd.DataFrame([profile])
    pred_idx = int(pipeline.predict(input_df)[0])
    pred_emotion = str(label_encoder.inverse_transform([pred_idx])[0])

    probabilities: dict[str, float] = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        for cls_name, prob in zip(label_encoder.classes_, probs):
            probabilities[str(cls_name)] = round(float(prob) * 100.0, 2)

    sorted_probs = dict(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "model_used": bundle["model_name"],
        "predicted_emotion": pred_emotion,
        "confidence_scores": sorted_probs,
    }


def prompt_interactive() -> dict[str, Any]:
    """Prompt the user interactively in the terminal for profile features."""
    print("\n=== Social Media Emotional Well-Being Interactive Predictor ===")
    age = float(input("Enter Age (15-70) [default 25]: ") or 25)
    gender = (
        input(f"Enter Gender {VALID_GENDERS} [default Female]: ").strip() or "Female"
    )
    platform = (
        input(f"Enter Platform {VALID_PLATFORMS} [default Instagram]: ").strip()
        or "Instagram"
    )
    usage = float(input("Daily Usage Time in minutes (0-1440) [default 120]: ") or 120)
    posts = float(input("Posts Per Day (0-100) [default 3]: ") or 3)
    likes = float(input("Likes Received Per Day (0-5000) [default 65]: ") or 65)
    comments = float(input("Comments Received Per Day (0-1000) [default 18]: ") or 18)
    messages = float(input("Messages Sent Per Day (0-1000) [default 30]: ") or 30)

    return {
        "Age": age,
        "Gender": gender,
        "Platform": platform,
        "Daily_Usage_Time (minutes)": usage,
        "Posts_Per_Day": posts,
        "Likes_Received_Per_Day": likes,
        "Comments_Received_Per_Day": comments,
        "Messages_Sent_Per_Day": messages,
    }


def main() -> None:
    """Parse CLI arguments and output prediction results."""
    parser = argparse.ArgumentParser(
        description="Predict Dominant Emotion from Social Media Usage Metrics."
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive prompt mode.",
    )
    parser.add_argument(
        "--model",
        choices=["best", "rf", "mlp"],
        default="best",
        help="Select best model, RandomForest (rf), or MLP Neural Net (mlp).",
    )
    parser.add_argument("--age", type=float, default=24.0, help="User age (15-70)")
    parser.add_argument(
        "--gender", type=str, default="Female", help="Female, Male, or Non-binary"
    )
    parser.add_argument(
        "--platform", type=str, default="Instagram", help="Social media platform"
    )
    parser.add_argument(
        "--usage", type=float, default=95.0, help="Daily usage time in minutes"
    )
    parser.add_argument("--posts", type=float, default=4.0, help="Posts per day")
    parser.add_argument(
        "--likes", type=float, default=90.0, help="Likes received per day"
    )
    parser.add_argument(
        "--comments", type=float, default=25.0, help="Comments received per day"
    )
    parser.add_argument(
        "--messages", type=float, default=40.0, help="Messages sent per day"
    )

    args = parser.parse_args()

    try:
        if args.interactive:
            profile = prompt_interactive()
        else:
            profile = {
                "Age": args.age,
                "Gender": args.gender,
                "Platform": args.platform,
                "Daily_Usage_Time (minutes)": args.usage,
                "Posts_Per_Day": args.posts,
                "Likes_Received_Per_Day": args.likes,
                "Comments_Received_Per_Day": args.comments,
                "Messages_Sent_Per_Day": args.messages,
            }

        result = predict_single_profile(profile, model_choice=args.model)

        print("\n" + "=" * 55)
        print(f"Model Used        : {result['model_used']}")
        print(f"Predicted Emotion : {result['predicted_emotion']}")
        print("-" * 55)
        print("Class Confidence Breakdown:")
        for emotion, score in result["confidence_scores"].items():
            bar = "#" * int(score // 5)
            print(f"  {emotion:<12} : {score:>6.2f}%  {bar}")
        print("=" * 55 + "\n")

    except KeyboardInterrupt:
        print("\nPrediction cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
