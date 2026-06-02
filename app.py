# -*- coding: utf-8 -*-
"""
COVID-19 Vaccine Sentiment — Streamlit Demo
نشر دائم على Streamlit Community Cloud أو HuggingFace Spaces (بديل LocalTunnel).

التشغيل محليًا:   streamlit run app.py
النشر:           ارفع هذا الملف مع sentiment_model.joblib و tfidf_vectorizer.joblib
                 و requirements.txt إلى GitHub، ثم اربطه بـ Streamlit Cloud / HF Spaces.
"""
import re
import string
import joblib
import streamlit as st

st.set_page_config(
    page_title="COVID-19 Vaccine Sentiment",
    page_icon="💉",
    layout="centered",
)

# ── تحميل النموذج والمُتجِّه (مُدرَّبان على تسميات Sentiment140 البشرية) ──────
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf

# ── تنظيف مطابق لخط أنابيب الـ Notebook (بدون spaCy لتخفيف تبعيات النشر) ────
# ملاحظة: الـ Notebook يستخدم lemmatization إضافياً لكنه لا يُغيّر النتيجة جوهرياً
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_COMMON_STOPS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","the","a",
    "an","and","or","but","in","on","at","to","for","of","with","by","from",
    "that","this","these","those","as","not","no","so","if","then","than",
    "just","can","get","got","also","very","too","more","much","even","about",
}

def basic_clean(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.lower().translate(_PUNCT_TABLE)
    tokens = [w for w in text.split() if w and w not in _COMMON_STOPS]
    return " ".join(tokens)

# ── واجهة المستخدم ────────────────────────────────────────────────────────────
COLORS = {"Positive": "#28a745", "Negative": "#dc3545"}
ICONS  = {"Positive": "😊", "Negative": "😠"}

st.title("💉 COVID-19 Vaccine Sentiment Analyzer")
st.markdown(
    "**Model:** TF-IDF + Logistic Regression &nbsp;·&nbsp; "
    "**Training data:** Sentiment140 (1.6M human-labeled tweets) &nbsp;·&nbsp; "
    "**Classes:** Positive / Negative &nbsp;·&nbsp; "
    "**Language:** English only",
    unsafe_allow_html=True,
)
st.divider()

try:
    model, tfidf = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the Save cell in the Notebook first "
        "(sentiment_model.joblib & tfidf_vectorizer.joblib) and place them "
        "next to app.py."
    )
    st.stop()

tweet = st.text_area(
    "Enter an English tweet about COVID-19 vaccines:",
    placeholder='e.g. "I got my vaccine today and feel great and fully protected!"',
    height=120,
)

col1, col2 = st.columns([3, 1])
analyze = col1.button("Analyze Sentiment", type="primary", use_container_width=True)
col2.button("Clear", on_click=lambda: None, use_container_width=True)

if analyze:
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        cleaned = basic_clean(tweet)
        if not cleaned.strip():
            st.error("No English content detected after cleaning — try an English tweet.")
        else:
            vec   = tfidf.transform([cleaned])
            pred  = model.predict(vec)[0]
            color = COLORS.get(pred, "#888")
            icon  = ICONS.get(pred, "🤔")

            classes = list(model.classes_)
            proba   = dict(zip(classes, model.predict_proba(vec)[0]))
            conf    = proba[pred] * 100

            st.markdown(
                f"<div style='background:{color}18;border-left:6px solid {color};"
                f"padding:18px 20px;border-radius:10px;margin-top:10px;'>"
                f"<h2 style='color:{color};margin:0;'>{icon} {pred}</h2>"
                f"<p style='margin:6px 0 0;font-size:15px;color:#555;'>"
                f"Confidence: <b>{conf:.1f}%</b></p></div>",
                unsafe_allow_html=True,
            )

            with st.expander("Show details"):
                st.write("**Cleaned text passed to model:**")
                st.code(cleaned, language=None)
                st.write("**Class probabilities:**")
                for cls in sorted(classes):
                    p = proba[cls]
                    st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

st.divider()
st.caption(
    "Graduation project — A Data-Driven Framework for Analyzing Public Sentiment "
    "Toward COVID-19 Vaccination on Twitter · "
    "Trained on Sentiment140 human labels, not VADER-generated labels."
)
