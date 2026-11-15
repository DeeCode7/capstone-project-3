# Technical Specification (SPEC.md): Social Media Usage and Emotional Well-Being Classifier

## 1. System Overview
* **Problem Statement:** Multi-class supervised classification predicting `Dominant_Emotion` (`Happiness`, `Sadness`, `Anger`, `Anxiety`, `Boredom`, `Neutral`) from daily social media engagement metrics and user demographics.
* **Models Compared:** Classical ML (`RandomForestClassifier`) vs. Deep Learning (`MLPClassifier`).
* **Reproducibility:** Global random seed `42`.

## 2. Defensive Data Cleaning Rules
1. Strip whitespace and remove empty or duplicate header rows.
2. Detect and recover rows where `Age` and `Gender` values are transposed.
3. Coerce numeric columns (`Age`, `Daily_Usage_Time (minutes)`, `Posts_Per_Day`, `Likes_Received_Per_Day`, `Comments_Received_Per_Day`, `Messages_Sent_Per_Day`) and clip to valid domain bounds.
4. Filter target `Dominant_Emotion` strictly to the 6 canonical classes.
5. Fallback to a deterministic synthetic generator if local CSVs are absent.
