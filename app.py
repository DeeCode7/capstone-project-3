"""Streamlit Web UI for Social Media Usage and Emotional Well-Being Classification."""

import json
import pandas as pd
import streamlit as st

from src.cli import predict_single_profile
from src.config import (
    METRICS_PATH,
    VALID_GENDERS,
    VALID_PLATFORMS,
)
from src.train import train_and_evaluate

st.set_page_config(
    page_title="Social Media Emotion Classifier",
    page_icon="🧠",
    layout="wide",
)

EMOTION_BADGES: dict[str, str] = {
    "Happiness": "😊 Happiness",
    "Sadness": "😢 Sadness",
    "Anger": "😠 Anger",
    "Anxiety": "😰 Anxiety",
    "Boredom": "😐 Boredom",
    "Neutral": "😌 Neutral",
}


@st.cache_data
def load_metrics() -> dict:
    """Load saved evaluation metrics or trigger initial training if absent."""
    if not METRICS_PATH.exists():
        train_and_evaluate()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Render the Streamlit application."""
    st.title("🧠 Social Media Usage & Emotional Well-Being Classifier")
    st.markdown(
        "Predict dominant emotional states from daily social media activity using "
        "**Classical Machine Learning (Random Forest)** and **Deep Learning (MLP Neural Network)**."
    )

    metrics_data = load_metrics()

    with st.sidebar:
        st.header("⚙️ Model & Data Controls")
        model_label = st.selectbox(
            "Select Inference Engine",
            options=[
                ("best", f"Best Promoted ({metrics_data.get('best_model', 'Auto')})"),
                ("rf", "RandomForest Classifier (ML)"),
                ("mlp", "MLP Neural Network (DL)"),
            ],
            format_func=lambda x: x[1],
        )
        selected_model_key = model_label[0]

        st.divider()
        st.subheader("📊 Active Dataset Info")
        st.write(f"**Source:** {metrics_data.get('data_source', 'N/A')}")
        st.write(f"**Train + Val Rows:** {metrics_data.get('train_samples', 0)}")
        st.write(f"**Test Rows:** {metrics_data.get('test_samples', 0)}")

        if st.button("🔄 Retrain Models on data/ CSVs", use_container_width=True):
            with st.spinner("Cleaning CSVs and training ML/DL pipelines..."):
                train_and_evaluate()
                st.cache_data.clear()
                st.rerun()

    tab_predict, tab_benchmark = st.tabs(
        ["🔍 Live Emotion Predictor", "📈 ML vs. DL Model Benchmarks"]
    )

    with tab_predict:
        col_form, col_result = st.columns([1.1, 1], gap="large")

        with col_form:
            st.subheader("User Profile & Daily Activity")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=15, max_value=70, value=27)
            with c2:
                gender = st.selectbox("Gender", options=VALID_GENDERS, index=0)
            with c3:
                platform = st.selectbox("Platform", options=VALID_PLATFORMS, index=0)

            usage = st.slider(
                "Daily Usage Time (minutes)",
                min_value=10,
                max_value=300,
                value=145,
                step=5,
            )

            m1, m2 = st.columns(2)
            with m1:
                posts = st.number_input(
                    "Posts Per Day", min_value=0, max_value=50, value=5
                )
                likes = st.number_input(
                    "Likes Received Per Day", min_value=0, max_value=500, value=75
                )
            with m2:
                comments = st.number_input(
                    "Comments Received Per Day", min_value=0, max_value=200, value=25
                )
                messages = st.number_input(
                    "Messages Sent Per Day", min_value=0, max_value=200, value=32
                )

            predict_clicked = st.button(
                "Predict Dominant Emotion", type="primary", use_container_width=True
            )

        with col_result:
            st.subheader("Prediction Output")
            if predict_clicked or "last_result" not in st.session_state:
                profile = {
                    "Age": float(age),
                    "Gender": gender,
                    "Platform": platform,
                    "Daily_Usage_Time (minutes)": float(usage),
                    "Posts_Per_Day": float(posts),
                    "Likes_Received_Per_Day": float(likes),
                    "Comments_Received_Per_Day": float(comments),
                    "Messages_Sent_Per_Day": float(messages),
                }
                st.session_state["last_result"] = predict_single_profile(
                    profile, model_choice=selected_model_key
                )

            res = st.session_state["last_result"]
            emotion_label = EMOTION_BADGES.get(
                res["predicted_emotion"], res["predicted_emotion"]
            )

            st.metric(
                label=f"Predicted Dominant Emotion ({res['model_used']})",
                value=emotion_label,
            )

            st.markdown("**Class Probability Distribution (%)**")
            prob_df = pd.DataFrame(
                list(res["confidence_scores"].items()),
                columns=["Emotion", "Confidence (%)"],
            )
            st.bar_chart(prob_df.set_index("Emotion"), height=280)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with tab_benchmark:
        st.subheader("Model Comparison on Held-Out Test Split")
        models_dict = metrics_data.get("models", {})
        summary_rows = []
        for m_name, m_vals in models_dict.items():
            summary_rows.append(
                {
                    "Model": m_name,
                    "Accuracy": m_vals["accuracy"],
                    "Macro-F1": m_vals["macro_f1"],
                    "Weighted-F1": m_vals["weighted_f1"],
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
        )

        cols = st.columns(2)
        for idx, (m_name, m_vals) in enumerate(models_dict.items()):
            with cols[idx % 2]:
                st.markdown(f"**Confusion Matrix: {m_name}**")
                cm_df = pd.DataFrame(
                    m_vals["confusion_matrix"],
                    index=[f"Actual {l}" for l in m_vals["labels"]],
                    columns=[f"Pred {l}" for l in m_vals["labels"]],
                )
                st.dataframe(cm_df, use_container_width=True)


if __name__ == "__main__":
    main()
