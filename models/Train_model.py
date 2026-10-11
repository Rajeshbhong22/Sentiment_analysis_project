
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# File paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

# Load dataset
df = pd.read_csv("data/review_sentiment_sample.csv")

# Check required columns
print("Dataset columns:", df.columns.tolist())

required_columns = {"review", "sentiment"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'review' and 'sentiment' columns."
    )

# Clean data
df = df.dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
df = df[df["review"] != ""]

# Remove duplicate examples
df = df.drop_duplicates(subset=["review", "sentiment"])

# Display available labels
print("\nSentiment labels:")
print(df["sentiment"].value_counts())

if df["sentiment"].nunique() < 2:
    raise ValueError("At least two sentiment classes are required.")

# Convert labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
X = df["review"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Create model pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000, random_state=42),
    ),
])

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

# Save model and label encoder together
with open(MODEL_PATH, "wb") as file:
    pickle.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
        },
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("\nModel saved successfully:", MODEL_PATH)
