
import pickle
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

st.set_page_config(
    page_title="ReviewSense AI | Amazon Reviews",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CUSTOM UI DESIGN
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
}

[data-testid="stHeader"] {
    background: rgba(245, 247, 251, 0.95);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #151c3a, #343d78);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(32, 42, 85, 0.12);
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbe4ff;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #202747;
    margin: 12px 0 14px 0;
}

.info-card {
    background: white;
    border: 1px solid #e7eaf2;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(20, 30, 60, 0.035);
}

.info-label {
    font-size: 13px;
    color: #68718a;
    font-weight: 600;
    margin-bottom: 10px;
}

.info-value {
    font-size: 27px;
    color: #202747;
    font-weight: 800;
}

.info-note {
    font-size: 12px;
    color: #7c8498;
    margin-top: 5px;
}

.result-card {
    background: white;
    border: 1px solid #e5e9f2;
    border-radius: 18px;
    padding: 26px;
    margin-top: 18px;
    margin-bottom: 16px;
}

.positive {
    color: #087c53;
    background: #e6f8ef;
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: 800;
    display: inline-block;
}

.negative {
    color: #b42332;
    background: #ffebed;
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: 800;
    display: inline-block;
}

.neutral {
    color: #825b00;
    background: #fff5d8;
    padding: 12px 18px;
    border-radius: 12px;
    font-weight: 800;
    display: inline-block;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 43px;
    border: 1px solid #dce1ed;
}

div.stButton > button[kind="primary"] {
    background: #4f46e5;
    border: none;
    color: white;
}

div.stButton > button[kind="primary"]:hover {
    background: #3730a3;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 17px;
    border: 1px solid #e7eaf2;
    border-radius: 14px;
}

div[data-testid="stTabs"] button {
    font-weight: 650;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e7eaf2;
}

.footer {
    text-align: center;
    color: #7c8498;
    font-size: 12px;
    padding: 22px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    return saved_data["model"], saved_data["label_encoder"]


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_model.py first.")
    st.code("python train_model.py")
    st.stop()

try:
    model, label_encoder = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.exception(error)
    st.stop()


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_sentiment_style(sentiment):
    value = str(sentiment).strip().lower()

    if value in ("positive", "pos"):
        return "😊", "positive"

    if value in ("negative", "neg"):
        return "😞", "negative"

    if value in ("neutral", "neu"):
        return "😐", "neutral"

    return "🔎", "neutral"


def predict_review(review):
    prediction_id = model.predict([review])[0]

    prediction = label_encoder.inverse_transform(
        [int(prediction_id)]
    )[0]

    probabilities = model.predict_proba([review])[0]

    classifier = model.named_steps.get("classifier")

    if classifier is not None:
        class_ids = classifier.classes_.astype(int)
        class_labels = label_encoder.inverse_transform(class_ids)
    else:
        class_labels = label_encoder.inverse_transform(
            list(range(len(probabilities)))
        )

    probability_df = pd.DataFrame({
        "Sentiment": class_labels,
        "Probability": probabilities * 100,
    })

    confidence = float(max(probabilities)) * 100

    return prediction, confidence, probability_df


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🛍️ ReviewSense AI")
    st.caption("Amazon Review Intelligence")

    st.divider()

    st.markdown("### 📌 Project Overview")
    st.write("Analyze customer opinions using NLP and machine learning.")

    st.markdown("### ⚙️ Model Information")
    st.markdown("""
    - **Algorithm:** Logistic Regression
    - **Features:** TF-IDF
    - **Task:** Sentiment Classification
    - **Framework:** Scikit-learn
    - **Interface:** Streamlit
    """)

    st.divider()

    st.markdown("### 💡 How to use")
    st.write("1. Enter a customer review.")
    st.write("2. Click Analyze Review.")
    st.write("3. View the prediction and confidence.")
    st.write("4. Upload a CSV for bulk analysis.")

    st.info(
        "Predictions depend on the training dataset. "
        "Confidence does not guarantee correctness."
    )


# ==========================================
# 6. HERO HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🛍️ ReviewSense AI</h1>
    <p>Amazon Product Review Sentiment Analysis Dashboard</p>
    <p>Understand customer opinions with Natural Language Processing
    and Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 7. DASHBOARD SUMMARY
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    metric_card("MODEL", "TF-IDF", "Text feature extraction")

with col2:
    metric_card("ALGORITHM", "Logistic Regression", "Supervised learning")

with col3:
    metric_card("ANALYSIS", "Real-time", "Review-level predictions")


st.markdown("")

# ==========================================
# 8. APPLICATION TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "✍️ Single Review",
    "📂 Batch CSV Analysis",
    "ℹ️ About the Model",
])


def set_review(text):
    st.session_state["review_input"] = text

def clear_review():
    st.session_state["review_input"] = ""
    st.session_state.pop("single_result", None)
# ==========================================
# TAB 1: SINGLE REVIEW
# ==========================================
with tab1:
    st.markdown(
        '<div class="section-title">Analyze Customer Feedback</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter an Amazon product review to discover its predicted sentiment."
    )

    st.markdown("**Try an example review:**")

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        if st.button("😊 Positive example", use_container_width=True):
            st.session_state["review_input"] = (
                "Amazing product! The quality is excellent "
                "and it works perfectly. I love it."
            )

    with sample_col2:
        if st.button("😞 Negative example", use_container_width=True):
            st.session_state["review_input"] = (
                "Very disappointing product. Poor quality, "
                "does not work properly, and a waste of money."
            )

    with sample_col3:
        if st.button("😐 Neutral example", use_container_width=True):
            st.session_state["review_input"] = (
                "The product arrived yesterday. "
                "It has the features mentioned in the description."
            )

    review = st.text_area(
        "Your product review",
        key="review_input",
        placeholder="Example: The product is comfortable, useful, and worth the price...",
        height=145,
    )

    analyze_col, clear_col = st.columns([3, 1])

    with analyze_col:
        analyze_button = st.button(
            "🔍 Analyze Review",
            type="primary",
            use_container_width=True,
            key="analyze_single",
        )

    with clear_col:
        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_review,
        )

    if analyze_button:
        if not review.strip():
            st.warning("Please enter a review before analyzing.")
        else:
            with st.spinner("Analyzing review..."):
                prediction, confidence, probability_df = predict_review(
                    review.strip()
                )

                st.session_state["single_result"] = {
                    "review": review.strip(),
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "probability_df": probability_df,
                }

    result = st.session_state.get("single_result")

    if result:
        st.markdown("---")
        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True,
        )

        prediction = result["prediction"]
        confidence = result["confidence"]
        emoji, css_class = get_sentiment_style(prediction)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown("**Predicted Customer Sentiment**")
        st.markdown(
            f'<div class="{css_class}">{emoji} {prediction.upper()}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**Analyzed Review**")
        st.write(result["review"])

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Predicted Sentiment", prediction)

        with result_col2:
            st.metric("Model Confidence", f"{confidence:.2f}%")

        st.progress(min(max(confidence / 100, 0.0), 1.0))

        st.caption(
            "Confidence is the model's estimated probability for its "
            "prediction, not a guarantee that the prediction is correct."
        )

        st.markdown('</div>', unsafe_allow_html=True)

        chart_col, details_col = st.columns([3, 2])

        with chart_col:
            st.markdown("### Sentiment Probability")
            chart_df = result["probability_df"].set_index("Sentiment")
            chart_df.columns = ["Probability (%)"]
            st.bar_chart(chart_df, y="Probability (%)")

        with details_col:
            st.markdown("### Class Probabilities")
            display_df = result["probability_df"].copy()
            display_df["Probability"] = display_df["Probability"].map(
                lambda value: f"{value * 1:.2f}%"
            )
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
            )


# ==========================================
# TAB 2: BATCH CSV ANALYSIS
# ==========================================
with tab2:
    st.markdown(
        '<div class="section-title">Bulk Review Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a CSV containing a column named `review`. "
        "Your existing sentiment model will analyze each valid review."
    )

    uploaded_file = st.file_uploader(
        "Upload your review dataset",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            review_column = next(
                (
                    column for column in df.columns
                    if str(column).strip().lower() == "review"
                ),
                None,
            )

            if review_column is None:
                st.error(
                    "The uploaded CSV must contain a column named 'review'."
                )
            else:
                valid_mask = (
                    df[review_column].notna()
                    & df[review_column].astype(str).str.strip().ne("")
                )

                results = df.copy()
                results["predicted_sentiment"] = None
                results["confidence_percent"] = None

                valid_reviews = df.loc[
                    valid_mask, review_column
                ].astype(str).tolist()

                if valid_reviews:
                    predictions = model.predict(valid_reviews)

                    decoded_predictions = label_encoder.inverse_transform(
                        predictions.astype(int)
                    )

                    probabilities = model.predict_proba(valid_reviews)
                    confidences = probabilities.max(axis=1) * 100

                    results.loc[
                        valid_mask, "predicted_sentiment"
                    ] = decoded_predictions

                    results.loc[
                        valid_mask, "confidence_percent"
                    ] = [
                        round(float(value), 2)
                        for value in confidences
                    ]

                analyzed = results.loc[valid_mask].copy()
                total = len(analyzed)

                st.success(
                    f"Analysis complete: {total} valid reviews processed."
                )

                positive_count = analyzed[
                    "predicted_sentiment"
                ].astype(str).str.lower().isin(
                    ["positive", "pos"]
                ).sum()

                negative_count = analyzed[
                    "predicted_sentiment"
                ].astype(str).str.lower().isin(
                    ["negative", "neg"]
                ).sum()

                neutral_count = analyzed[
                    "predicted_sentiment"
                ].astype(str).str.lower().isin(
                    ["neutral", "neu"]
                ).sum()

                count1, count2, count3, count4 = st.columns(4)

                with count1:
                    st.metric("Total Analyzed", total)

                with count2:
                    st.metric("😊 Positive", int(positive_count))

                with count3:
                    st.metric("😞 Negative", int(negative_count))

                with count4:
                    st.metric("😐 Neutral", int(neutral_count))

                st.markdown("### 📋 Prediction Results")

                search_text = st.text_input(
                    "Filter results by review text or sentiment",
                    placeholder="Search for positive, negative, or a keyword...",
                    key="batch_search",
                )

                filtered = results.copy()

                if search_text.strip():
                    searchable = (
                        filtered[review_column].fillna("").astype(str)
                        + " "
                        + filtered["predicted_sentiment"].fillna("").astype(str)
                    )
                    filtered = filtered[
                        searchable.str.contains(
                            search_text.strip(),
                            case=False,
                            regex=False,
                        )
                    ]

                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True,
                )

                if total:
                    chart_col, confidence_col = st.columns(2)

                    with chart_col:
                        st.markdown("### Sentiment Distribution")

                        sentiment_counts = (
                            analyzed["predicted_sentiment"]
                            .value_counts()
                        )
                        st.bar_chart(sentiment_counts)

                    with confidence_col:
                        st.markdown("### Confidence Summary")

                        confidence_values = pd.to_numeric(
                            analyzed["confidence_percent"],
                            errors="coerce",
                        ).dropna()

                        if not confidence_values.empty:
                            st.metric(
                                "Average Confidence",
                                f"{confidence_values.mean():.2f}%",
                            )
                            st.metric(
                                "Highest Confidence",
                                f"{confidence_values.max():.2f}%",
                            )

                output = BytesIO()

                results.to_csv(
                    output,
                    index=False,
                    encoding="utf-8-sig",
                )

                st.download_button(
                    label="⬇️ Download Complete Results as CSV",
                    data=output.getvalue(),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Unable to process this CSV: {error}")


# ==========================================
# TAB 3: ABOUT THE MODEL
# ==========================================
with tab3:
    st.markdown(
        '<div class="section-title">About ReviewSense AI</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "ReviewSense AI is a sentiment classification application "
        "designed to analyze customer feedback from product reviews."
    )

    st.markdown("### How the application works")

    st.markdown("""
    1. **Input:** A customer review is entered or uploaded from a CSV.
    2. **Preprocessing and features:** TF-IDF converts text into numerical
       features according to the fitted training pipeline.
    3. **Classification:** Logistic Regression predicts a sentiment class.
    4. **Probability:** The model provides class probabilities when supported.
    5. **Output:** The application displays sentiment, confidence, and charts.
    """)

    st.markdown("### Technologies used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("**Python**")
        st.caption("Programming language")

    with tech2:
        st.markdown("**Pandas**")
        st.caption("Data processing")

    with tech3:
        st.markdown("**Scikit-learn**")
        st.caption("Machine learning")

    with tech4:
        st.markdown("**Streamlit**")
        st.caption("Web dashboard")

    st.warning(
        "The model can misclassify sarcasm, mixed opinions, spelling errors, "
        "or reviews that differ from the training dataset."
    )


# ==========================================
# 9. FOOTER
# ==========================================
st.divider()

st.markdown("""
<div class="footer">
    ReviewSense AI · Amazon Product Review Sentiment Analysis<br>
    Built with Python, TF-IDF, Logistic Regression, Pandas and Streamlit
</div>
""", unsafe_allow_html=True)
